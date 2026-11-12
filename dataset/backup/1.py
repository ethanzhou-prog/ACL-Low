import json
import os
import re
from pathlib import Path

# 必须在导入 vLLM 之前指定 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


BASE_MODEL_PATH = "/data/chaoni/zzy/Qwen2.5-7B"
DATA_PATH = Path("/data/chaoni/zzy/ACL-Low/dataset")
OUTPUT_DIR = Path(".")

TEST_SETS = {
    "java_to_arkts": DATA_PATH / "codenet-1000-demo.jsonl",
}

NUM_GENERATIONS = 8
MAX_INPUT_LEN = 2048
MAX_NEW_TOKENS = 2048
MAX_MODEL_LEN = MAX_INPUT_LEN + MAX_NEW_TOKENS


def build_prompt(instruction: str, input_code: str) -> str:
    """Construct a Java-to-ArkTS translation prompt."""

    return (
        f"Instruction: {instruction.strip()}\n"
        "Requirement: Return only compilable ArkTS source code. "
        "Do not include explanations, Markdown code fences, examples, "
        "or additional tasks.\n"
        f"Input Code:\n{input_code.strip()}\n"
        "ArkTS Code:\n"
    )


def clean_arkts_output(text: str) -> str:
    """Extract generated code without checking language or correctness."""

    text = text.strip()

    # 有 Markdown 代码块时提取内容，不限制语言标签。
    match = re.search(
        r"```[^\r\n]*\r?\n(.*?)```",
        text,
        flags=re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    # 没有代码块时保留完整生成内容，避免误删代码。
    return text


def load_test_data(file_path: Path, tokenizer):
    """Load JSONL records and construct prompts."""

    prompts = []
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {file_path}, line {line_number}"
                ) from exc

            instruction = obj["instruction"].strip()
            input_code = obj["input"].strip()
            target_code = obj.get("output", "").strip()
            label = obj.get("label", "")

            prompt = build_prompt(
                instruction=instruction,
                input_code=input_code,
            )

            token_ids = tokenizer.encode(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_INPUT_LEN,
            )

            truncated_prompt = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            )

            prompts.append(truncated_prompt)
            records.append(
                {
                    "instruction": instruction,
                    "input": input_code,
                    "output": target_code,
                    "label": label,
                }
            )

    return prompts, records


def save_txt_results(
    output_path: Path,
    records: list,
    predictions: list,
) -> None:
    """Save one generated candidate per line."""

    with output_path.open("w", encoding="utf-8") as file:
        for record, candidates in zip(records, predictions):
            label = record["label"]

            for sample_id, candidate in enumerate(candidates, start=1):
                escaped_code = candidate.replace("\n", "\\n")

                file.write(
                    f"{label}\t{sample_id}\t{escaped_code}\n"
                )


def save_jsonl_results(
    output_path: Path,
    records: list,
    predictions: list,
) -> None:
    """Save generated candidates in JSONL format."""

    with output_path.open("w", encoding="utf-8") as file:
        for record, candidates in zip(records, predictions):
            output_record = {
                "instruction": record["instruction"],
                "input": record["input"],
                "output": record["output"],
                "label": record["label"],
                "model_outputs": candidates,
            }

            file.write(
                json.dumps(output_record, ensure_ascii=False) + "\n"
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
    )

    print(">>> Loading vLLM engine...")

    llm = LLM(
        model=BASE_MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        max_num_seqs=8,
        disable_custom_all_reduce=True,
    )

    sampling_params = SamplingParams(
        n=NUM_GENERATIONS,
        temperature=0,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
        skip_special_tokens=True,
        stop=[
            "\nInstruction:",
            "\nInstructions:",
            "\nRequirement:",
            "\nInput Code:",
            "\nOutput Code:",
            "\nExplanation:",
            "\nIn this task",
            "\nHere are",
            "\nYou can use",
            "\nI translated",
            "\nThe Java code",
            "\nThe translated code",
        ],
    )

    for task, file_path in TEST_SETS.items():
        print(f"\n=== Generating for task: {task} ===")

        prompts, records = load_test_data(
            file_path=file_path,
            tokenizer=tokenizer,
        )

        print(f">>> Loaded {len(prompts)} prompts")
        print(
            f">>> Generating {NUM_GENERATIONS} candidates "
            "per prompt..."
        )

        request_outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
        )

        predictions = []

        for request_output in request_outputs:
            candidates = []

            for candidate in request_output.outputs:
                # 保留每个候选，不检测语言、不过滤错误或空输出。
                cleaned_code = clean_arkts_output(candidate.text)
                candidates.append(cleaned_code)

            predictions.append(candidates)

        txt_path = (
            OUTPUT_DIR / f"codenet-1000_{task}.txt"
        )
        jsonl_path = (
            OUTPUT_DIR / f"codenet-1000_{task}.jsonl"
        )

        save_txt_results(
            output_path=txt_path,
            records=records,
            predictions=predictions,
        )

        save_jsonl_results(
            output_path=jsonl_path,
            records=records,
            predictions=predictions,
        )

        print(
            f">>> Saved candidates: "
            f"{sum(len(items) for items in predictions)}"
        )
        print(f">>> Saved TXT: {txt_path}")
        print(f">>> Saved JSONL: {jsonl_path}")

    print("\n>>> Generation completed.")


if __name__ == "__main__":
    main()
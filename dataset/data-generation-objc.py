import json
import os
import re
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


BASE_MODEL_PATH = "/data/chaoni/zzy/Qwen2.5-7B"
DATA_PATH = Path("/data/chaoni/zzy/ACL-Low/dataset")
OUTPUT_DIR = Path(".")

TEST_SETS = {
    "java_to_arkts": DATA_PATH / "codenet-full-data.jsonl",
}

NUM_GENERATIONS = 8
MAX_INPUT_LEN = 2048
MAX_NEW_TOKENS = 2048
MAX_MODEL_LEN = MAX_INPUT_LEN + MAX_NEW_TOKENS


def build_prompt(instruction: str, input_code: str) -> str:
    return (
        f"Instruction: {instruction}\n"
        "Return only compilable Objective-C source code.\n"
        "Do not include explanations, Markdown fences, examples, or additional tasks.\n"
        f"Input Code:\n{input_code}\n"
        "Objective-C Code:\n"
    )


def clean_objective_c_output(text: str) -> str:
    """Extract Objective-C code and remove explanations or continued tasks."""

    text = text.strip()

    # Prefer a fenced code block when the model produced one.
    fenced_blocks = re.findall(
        r"```(?:objective-c|objectivec|objc|c)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in fenced_blocks:
        if any(
            marker in block
            for marker in (
                "#import",
                "@interface",
                "@implementation",
                "@autoreleasepool",
            )
        ):
            text = block.strip()
            break

    # Remove leading natural-language text.
    starts = [
        position
        for marker in (
            "#import",
            "@interface",
            "@implementation",
            "int main",
        )
        if (position := text.find(marker)) >= 0
    ]
    if starts:
        text = text[min(starts):]

    # Remove explanations or additional generated tasks after the code.
    stop_markers = (
        "\n```",
        "\nInstruction:",
        "\nInstructions:",
        "\nInput Code:",
        "\nOutput Code:",
        "\nObjective-C Code:",
        "\nExplanation:",
        "\nIn this task",
        "\nHere are the guidelines",
        "\nHere's the translated code",
        "\nYou can use the following",
        "\nI translated the Java code",
        "\nThe given Java code",
    )

    cut_positions = [
        position
        for marker in stop_markers
        if (position := text.find(marker)) >= 0
    ]
    if cut_positions:
        text = text[:min(cut_positions)]

    text = text.strip()

    # Reject outputs that do not look like Objective-C.
    objc_markers = (
        "#import",
        "@interface",
        "@implementation",
        "@autoreleasepool",
        "Foundation/Foundation.h",
    )
    if not any(marker in text for marker in objc_markers):
        return ""

    return text


def load_test_data(file_path: Path, tokenizer):
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

            prompt = build_prompt(instruction, input_code)

            token_ids = tokenizer.encode(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_INPUT_LEN,
            )
            prompt = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            )

            prompts.append(prompt)
            records.append(
                {
                    "instruction": instruction,
                    "input": input_code,
                    "output": obj.get("output", "").strip(),
                    "label": obj.get("label", ""),
                }
            )

    return prompts, records


def save_txt_results(
    output_path: Path,
    records: list,
    predictions: list,
) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        for record, candidates in zip(records, predictions):
            for sample_id, candidate in enumerate(candidates, start=1):
                escaped_code = candidate.replace("\n", "\\n")
                file.write(
                    f"{record['label']}\t{sample_id}\t{escaped_code}\n"
                )


def save_jsonl_results(
    output_path: Path,
    records: list,
    predictions: list,
) -> None:
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
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        max_num_seqs=64,
        disable_custom_all_reduce=True,
    )

    sampling_params = SamplingParams(
        n=NUM_GENERATIONS,
        temperature=0.5,
        top_p=0.95,
        max_tokens=MAX_NEW_TOKENS,
        skip_special_tokens=True,
        stop=[
            "\nInstruction:",
            "\nInstructions:",
            "\nInput Code:",
            "\nExplanation:",
            "\nIn this task",
            "\nYou can use the following",
            "\nI translated the Java code",
            "\nThe given Java code",
        ],
    )

    for task, file_path in TEST_SETS.items():
        print(f"\n=== Generating for task: {task} ===")

        prompts, records = load_test_data(
            file_path=file_path,
            tokenizer=tokenizer,
        )

        print(f">>> Loaded {len(prompts)} prompts")
        print(f">>> Generating {NUM_GENERATIONS} candidates per prompt...")

        request_outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
        )

        predictions = []
        invalid_count = 0

        for request_output in request_outputs:
            candidates = []

            for candidate in request_output.outputs:
                cleaned_code = clean_objective_c_output(candidate.text)

                if cleaned_code:
                    candidates.append(cleaned_code)
                else:
                    invalid_count += 1

            predictions.append(candidates)

        txt_path = OUTPUT_DIR / f"codenet-full-data{task}.txt"
        jsonl_path = OUTPUT_DIR / f"codenet-full-data{task}.jsonl"

        save_txt_results(txt_path, records, predictions)
        save_jsonl_results(jsonl_path, records, predictions)

        print(f">>> Invalid non-Objective-C outputs removed: {invalid_count}")
        print(f">>> Saved TXT: {txt_path}")
        print(f">>> Saved JSONL: {jsonl_path}")

    print("\n>>> Generation completed.")


if __name__ == "__main__":
    main()
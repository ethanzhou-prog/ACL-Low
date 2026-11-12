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
        "Requirement: Return only compilable TypeScript source code. "
        "Do not include explanations, Markdown code fences, examples, "
        "or additional tasks.\n"
        f"Input Code:\n{input_code.strip()}\n"
        "ArkTS Code:\n"
    )


def clean_arkts_output(text: str) -> str:
    """Extract ArkTS code and remove explanations or Markdown."""

    text = text.strip()

    if not text:
        return ""

    # 如果模型使用了 Markdown 代码块，优先提取其中的代码。
    fenced_blocks = re.findall(
        r"```(?:arkts|typescript|ts|javascript|js)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced_blocks:
        for block in fenced_blocks:
            if looks_like_arkts(block):
                text = block.strip()
                break
        else:
            text = fenced_blocks[0].strip()

    # 删除代码前面的自然语言。
    code_start_markers = (
        "@Entry",
        "@Component",
        "import ",
        "export ",
        "function ",
        "class ",
        "interface ",
        "enum ",
        "type ",
        "namespace ",
        "struct ",
        "const ",
        "let ",
        "var ",
        "for (",
        "while (",
    )

    start_positions = [
        position
        for marker in code_start_markers
        if (position := text.find(marker)) >= 0
    ]

    if start_positions:
        text = text[min(start_positions):]

    # 删除代码后面的解释、额外任务或重复提示词。
    stop_markers = (
        "\n```",
        "\nInstruction:",
        "\nInstructions:",
        "\nRequirement:",
        "\nInput Code:",
        "\nOutput Code:",
        "\nArkTS Code:",
        "\nExplanation:",
        "\nIn this task",
        "\nHere are",
        "\nYou can use",
        "\nI translated",
        "\nThe Java code",
        "\nThe translated code",
        "\nThis code",
    )

    stop_positions = [
        position
        for marker in stop_markers
        if (position := text.find(marker)) >= 0
    ]

    if stop_positions:
        text = text[:min(stop_positions)]

    text = text.strip()

    # 清除可能残留的 Markdown 标记。
    text = re.sub(
        r"^```(?:arkts|typescript|ts|javascript|js)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text)

    text = text.strip()

    if not looks_like_arkts(text):
        return ""

    return text


def looks_like_arkts(text: str) -> bool:
    """Check whether generated text resembles ArkTS/TypeScript code."""

    text = text.strip()

    if not text:
        return False

    # class Main 不会被 Ark 作为程序入口自动执行。
    if re.search(r"\bclass\s+Main\b", text):
        return False

    # 这些语法不属于 ArkTS/TypeScript，出现时说明仍残留 Java。
    invalid_arkts_patterns = (
        r"\bpublic\s+static\s+void\s+main\s*\(",
        r"\bstatic\s+void\s+main\s*\(",
        r"\bSystem\.(?:out|err)\s*\.",
        r"\bString\s*(?:\[\s*\]|\.\.\.)",
        r"\bnew\s+Scanner\s*\(",
        r"\bScanner\s+[A-Za-z_$][\w$]*",
        r"\bimport\s+java\.",
        r"(?m)^\s*package\s+[\w.]+\s*;",
        r"\b(?:Integer|Long|Double|Float|Boolean)\.parse\w*\s*\(",
        r"\b(?:ArrayList|LinkedList|HashMap|HashSet|TreeMap|"
        r"TreeSet|PriorityQueue)\s*<",
        r"\b(?:public|private|protected)\s+"
        r"(?:(?:static|final|abstract|synchronized)\s+)*"
        r"(?:void|boolean|byte|short|int|long|float|double|char|String)"
        r"\s+[A-Za-z_$][\w$]*",
        r"\bstatic\s+final\s+"
        r"(?:boolean|byte|short|int|long|float|double|char|String)\b",
        r"\bstatic\s+const\b",
    )

    if any(
        re.search(pattern, text)
        for pattern in invalid_arkts_patterns
    ):
        return False

    # 只使用具有 ArkTS/TypeScript 区分度的标记。
    # 不再使用 class、import、for、while 等 Java 共有标记。
    arkts_patterns = (
        r"\b(?:const|let|var)\s+[A-Za-z_$][\w$]*",
        r"\bfunction\s+[A-Za-z_$][\w$]*\s*\(",
        r"\b(?:interface|type|enum|namespace)\s+"
        r"[A-Za-z_$][\w$]*",
        r"\bconsole\.(?:log|error|warn)\s*\(",
        r"\breadline\s*\(",
        r"\b[A-Za-z_$][\w$]*\s*:\s*"
        r"(?:number|string|boolean|bigint|unknown|"
        r"[A-Za-z_$][\w$]*(?:\[\])?)",
        r"=>",
    )

    return any(
        re.search(pattern, text)
        for pattern in arkts_patterns
    )


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
        max_num_seqs=64,
        disable_custom_all_reduce=True,
    )

    sampling_params = SamplingParams(
        n=NUM_GENERATIONS,
        temperature=0.1,
        top_p=0.95,
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
            f">>> Generating {NUM_GENERATIONS} ArkTS candidates "
            "per prompt..."
        )

        request_outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
        )

        predictions = []
        invalid_count = 0

        for request_output in request_outputs:
            candidates = []

            for candidate in request_output.outputs:
                cleaned_code = clean_arkts_output(
                    candidate.text
                )

                if cleaned_code:
                    candidates.append(cleaned_code)
                else:
                    invalid_count += 1

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
            f">>> Invalid or non-ArkTS outputs removed: "
            f"{invalid_count}"
        )
        print(f">>> Saved TXT: {txt_path}")
        print(f">>> Saved JSONL: {jsonl_path}")

    print("\n>>> Generation completed.")


if __name__ == "__main__":
    main()
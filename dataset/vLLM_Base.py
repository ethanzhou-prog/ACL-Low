# import json
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# from tqdm import tqdm
# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams

# base_model_path = "/data/chaoni/zzy/Qwen2.5-7B"
# data_path = "/data/chaoni/zzy/ACL-Low/dataset"

# test_sets = {
#     "java_to_objc": os.path.join(data_path, "codenet-1000-demo.jsonl"),
# }

# MAX_INPUT_LEN = 2048
# MAX_NEW_TOKENS = 2048
# if __name__ == '__main__':
#   print(">>> loading tokenizer...")
#   tokenizer = AutoTokenizer.from_pretrained(
#     base_model_path,
#     trust_remote_code=True,
#     extra_special_tokens={}
# )

#   print(">>> loading vLLM engine...")
#   llm = LLM(
#     model=base_model_path,
#     trust_remote_code=True,
#     tensor_parallel_size=1,
#     max_model_len=4096,
#     gpu_memory_utilization=0.9,
#     enforce_eager=True,
#     max_num_seqs=32,
#     disable_custom_all_reduce=True,
# )

#   sampling_params = SamplingParams(
#       temperature=0.0,
#       max_tokens=MAX_NEW_TOKENS,
#       skip_special_tokens=True,
#   )

#   for task, file_path in test_sets.items():
#       print(f"\n=== Generating for task: {task} ===")

#       prompts = []
#       orig_instructions, orig_inputs, orig_outputs, orig_labels = [], [], [], []

#       with open(file_path, "r", encoding="utf-8") as f:
#           for line in f:
#               obj = json.loads(line)

#               inst = obj["instruction"].strip()
#               inp = obj["input"].strip()
#               tgt = obj["output"].strip()
#               label = obj.get("label", "")

#               prompt = f"Instruction: {inst}\nInput Code:\n{inp}\nOutput Code:\n"

#             #  用 tokenizer 截断，但仍返回字符串
#               token_ids = tokenizer.encode(
#                   prompt,
#                   truncation=True,
#                   max_length=MAX_INPUT_LEN
#               )
#               prompt = tokenizer.decode(token_ids, skip_special_tokens=True)

#               prompts.append(prompt)

#               orig_instructions.append(inst)
#               orig_inputs.append(inp)
#               orig_outputs.append(tgt)
#               orig_labels.append(label)

#       print(">>> Generating with vLLM...")

#     # 已经去掉了 lora_request 参数，直接传入 prompts 即可
#       outputs = llm.generate(prompts, sampling_params=sampling_params)

#       preds = [output.outputs[0].text.strip() for output in outputs]

#       txt_path = f"./Qwen-batch1{task}.txt"
#       with open(txt_path, "w", encoding="utf-8") as fw:
#           for pred in preds:
#               fw.write(pred.replace("\n", "\\n") + "\n")

#       jsonl_path = f"./Qwen-batch1{task}.jsonl"
#       with open(jsonl_path, "w", encoding="utf-8") as fw_jsonl:
#           for inst, inp, tgt, label, pred in zip(
#               orig_instructions, orig_inputs, orig_outputs, orig_labels, preds
#           ):
#               out_obj = {
#                   "instruction": inst,
#                   "input": inp,
#                   "output": tgt,
#                   "label": label,
#                   "model_output": pred,
#               }
#               fw_jsonl.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

#       print(f">>> Saved: {txt_path}")
#       print(f">>> Saved: {jsonl_path}")

import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


base_model_path = "/data/chaoni/zzy/Qwen2.5-7B"
data_path = "/data/chaoni/zzy/ACL-Low/dataset"

test_sets = {
    "java_to_objc": os.path.join(
        data_path,
        "codenet-1000-demo.jsonl",
    ),
}

MAX_INPUT_LEN = 2048
MAX_NEW_TOKENS = 2048

END_MARKER = "</OBJECTIVE_C>"


def build_prompt(tokenizer, instruction, input_code):
    prefix = (
        f"Instruction: {instruction}\n"
        "Translate the following Java code into Objective-C.\n"
        "Generate only the translated Objective-C source code.\n"
        "Do not generate Markdown, explanations, notes, Java code, "
        "or any text outside the Objective-C translation.\n"
        f"End the translation with {END_MARKER}.\n"
        "Input Java Code:\n"
    )

    # Prompt 以输出区域开头标记结束，模型从这里直接续写 Objective-C
    suffix = "\nOutput Objective-C Code:\n<OBJECTIVE_C>\n"

    prefix_ids = tokenizer.encode(
        prefix,
        add_special_tokens=False,
    )
    suffix_ids = tokenizer.encode(
        suffix,
        add_special_tokens=False,
    )

    available_input_tokens = (
        MAX_INPUT_LEN
        - len(prefix_ids)
        - len(suffix_ids)
    )

    if available_input_tokens <= 0:
        raise ValueError("MAX_INPUT_LEN 太小，无法容纳提示词")

    # 只截断 Java 输入，不会截掉 Objective-C 输出要求
    input_ids = tokenizer.encode(
        input_code,
        add_special_tokens=False,
    )[:available_input_tokens]

    truncated_input = tokenizer.decode(
        input_ids,
        skip_special_tokens=True,
    )

    return prefix + truncated_input + suffix


def extract_translation(generated_text):
    """
    Prompt 已经包含 <OBJECTIVE_C>，
    因此生成内容本身就是 Objective-C 翻译结果。
    """
    generated_text = generated_text.strip()

    # 防止模型重复输出开始标记
    if generated_text.startswith("<OBJECTIVE_C>"):
        generated_text = generated_text[len("<OBJECTIVE_C>"):].lstrip()

    # 正常情况下 vLLM 会在结束标记处停止，此处作为额外保护
    if END_MARKER in generated_text:
        generated_text = generated_text.split(END_MARKER, 1)[0]

    return generated_text.strip()


if __name__ == "__main__":
    print(">>> Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        extra_special_tokens={},
    )

    print(">>> Loading vLLM engine...")

    llm = LLM(
        model=base_model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        max_num_seqs=32,
        disable_custom_all_reduce=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        skip_special_tokens=True,

        # 唯一停止条件：Objective-C 翻译区域结束
        stop=[END_MARKER],
    )

    for task, file_path in test_sets.items():
        print(f"\n=== Generating for task: {task} ===")

        prompts = []
        orig_instructions = []
        orig_inputs = []
        orig_outputs = []
        orig_labels = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line in tqdm(file, desc="Preparing prompts"):
                if not line.strip():
                    continue

                obj = json.loads(line)

                instruction = obj["instruction"].strip()
                input_code = obj["input"].strip()
                target_code = obj["output"].strip()
                label = obj.get("label", "")

                prompt = build_prompt(
                    tokenizer,
                    instruction,
                    input_code,
                )

                prompts.append(prompt)
                orig_instructions.append(instruction)
                orig_inputs.append(input_code)
                orig_outputs.append(target_code)
                orig_labels.append(label)

        print(">>> Generating with vLLM...")

        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
        )

        # 此处只取得被 <OBJECTIVE_C> 区间限定的翻译结果
        predictions = [
            extract_translation(output.outputs[0].text)
            for output in outputs
        ]

        txt_path = f"./Qwen-batch1{task}.txt"

        with open(txt_path, "w", encoding="utf-8") as file:
            for prediction in predictions:
                file.write(
                    prediction.replace("\n", "\\n") + "\n"
                )

        jsonl_path = f"./Qwen-batch1{task}.jsonl"

        with open(jsonl_path, "w", encoding="utf-8") as file:
            for instruction, input_code, target_code, label, prediction in zip(
                orig_instructions,
                orig_inputs,
                orig_outputs,
                orig_labels,
                predictions,
            ):
                output_object = {
                    "instruction": instruction,
                    "input": input_code,
                    "output": target_code,
                    "label": label,
                    "model_output": prediction,
                }

                file.write(
                    json.dumps(
                        output_object,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(f">>> Saved: {txt_path}")
        print(f">>> Saved: {jsonl_path}")
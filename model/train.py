import os
from config import parse_args


def main():
    args = parse_args()
    from run_logging import run_logging
    with run_logging(args):
        run(args)


def run(args):
    # Parse once, before importing torch/transformers, so config controls CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["HF_HOME"] = args.hf_home

    if args.check_data:
        from data_loader import load_stage_dataset
        for stage in ("dae", "sft", "dpo"):
            if not getattr(args, "skip_" + stage):
                print(f"{stage}: {len(load_stage_dataset(args, None, stage))} tasks")
                print(f"{stage}/val: {len(load_stage_dataset(args, None, stage, split='val'))} tasks")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    from train_dae import train_dae
    from train_sft import train_sft
    from train_dpo import train_dpo

    from importlib.metadata import version
    for package in ('torch', 'transformers', 'peft', 'accelerate'):
        print(f'[Environment] {package}={version(package)}')
    print(f'[Environment] CUDA_VISIBLE_DEVICES={args.gpu_id}; CUDA available={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(f'[Environment] Visible GPU {index}: {props.name}; VRAM={props.total_memory / 1024**3:.2f} GiB')

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Single GPU entry point; do not launch with torchrun")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_length

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code,
        torch_dtype=getattr(torch, args.dtype),
    )
    adapter_path = args.resume_from_checkpoint or args.trainer_checkpoint
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules=args.target_modules,
            bias="none", task_type="CAUSAL_LM",
        ))
    model.to(args.device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    if not args.skip_dae:
        print('[Stage] DAE started')
        train_dae(args, model, tokenizer)
        print('[Stage] DAE completed')

    if not args.skip_sft:
        print('[Stage] SFT started')
        train_sft(args, model, tokenizer)
        print('[Stage] SFT completed')

    if not args.skip_dpo:
        print('[Stage] DPO started')
        train_dpo(args, model, tokenizer)
        print('[Stage] DPO completed')

    print("All stages completed.")


if __name__ == "__main__":
    main()

import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Pipeline (DAE -> Structure-Weighted MR-SFT -> DPO)")

    # Environment
    parser.add_argument("--gpu_id", type=str, default="5")
    parser.add_argument("--hf_home", type=str, default="/data/chaoni/zzy/huggingface")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    # parser.add_argument("--seed", type=int, default=42)

    # Paths: relative dataset filenames are resolved against data_dir.
    parser.add_argument("--model_path", type=str, default="/data/chaoni/zzy/Qwen2.5-7B")
    parser.add_argument("--data_dir", type=str, default="/data/chaoni/zzy/ACL-Low/model/data")
    parser.add_argument("--output_dir", type=str, default="/data/chaoni/zzy/ACL-Low/model_out/qwen2.5-7b-demo")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Detailed run logs; default: output_dir/logs")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")

    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Load existing LoRA weights and start a new optimizer (legacy behavior)")
    parser.add_argument("--trainer_checkpoint", type=str, default=None,
                        help="Restore optimizer/step/RNG too; enable exactly one stage")

                        
    parser.add_argument("--dae_files", nargs="+", default=["dae_objc_train.jsonl"])
    parser.add_argument("--sft_files", nargs="+", default=["mr-sft-train.jsonl"])
    parser.add_argument("--dpo_files", nargs="+", default=["dpo-objc-train.jsonl"])

    parser.add_argument("--dae_val_files", nargs="+", default=["dae_objc_val.jsonl"])
    parser.add_argument("--sft_val_files", nargs="+", default=["mr-sft-val.jsonl"])
    parser.add_argument("--dpo_val_files", nargs="+", default=["dpo-objc-val.jsonl"])

    # Model / LoRA
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    #["q_proj", "k_proj", "v_proj", "o_proj"]

    # Shared training parameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_strategy", choices=["epoch", "steps"], default="epoch")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)

    # Epochs for independent stages
    parser.add_argument("--dae_epochs", type=int, default=5)
    parser.add_argument("--sft_epochs", type=int, default=5)
    parser.add_argument("--dpo_epochs", type=int, default=5)

    # Structure-Weighted MR-SFT
    parser.add_argument("--weight_mode", choices=["uniform", "java_objc_ast"], default="java_objc_ast")
    parser.add_argument("--max_references", type=int, default=4,
                        help="Maximum correct references per Java, selected by ascending structure_distance")
    
    parser.add_argument("--tau", type=float, default=2.0)
    
    parser.add_argument("--length_normalize", action="store_true",
                        help="Use token-mean SFT loss instead of screenshot sequence-sum objective")

    # DPO
    parser.add_argument("--dpo_learning_rate", type=float, default=5e-6)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--semantic_weight", type=float, default=1.0)
    parser.add_argument("--syntax_weight", type=float, default=0.25)

    # Stage switches: use skip flags to test one stage at a time.
    parser.add_argument("--skip_dae", action="store_true")
    parser.add_argument("--skip_sft", action="store_true")
    parser.add_argument("--skip_dpo", action="store_true")
    parser.add_argument("--check_data", action="store_true", help="Only validate data for enabled stages")

    args = parser.parse_args()
    enabled = [s for s in ("dae", "sft", "dpo") if not getattr(args, "skip_" + s)]
    if not enabled:
        parser.error("Enable at least one stage")
    if args.trainer_checkpoint and (len(enabled) != 1 or args.resume_from_checkpoint):
        parser.error("trainer_checkpoint requires exactly one stage and no resume_from_checkpoint")
    if "," in args.gpu_id:
        parser.error("This entry point supports one GPU")
    if args.tau < 0 or args.dpo_beta <= 0:
        parser.error("tau must be nonnegative and dpo_beta positive")
    if args.max_references < 1:
        parser.error("max_references must be positive")
    if any(getattr(args, s + "_epochs") <= 0 for s in enabled):
        parser.error("epochs must be positive")
    if args.device == "cpu" and args.dtype != "float32":
        parser.error("Use --dtype float32 on CPU")
    return args
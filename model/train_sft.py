import torch
from transformers import Trainer, TrainingArguments
from tokenize_utils import Collator
from utils import sequence_logps
from data_loader import load_stage_dataset
from run_logging import make_training_log_callback


class SFTTrainer(Trainer):
    def __init__(self, *args, stage, objective, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage, self.objective = stage, objective
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        # w_ij = softmax_j(-tau * D_AST(Java_i, ObjC_ij)), prepared by data_loader.
        # L = -(1/N) sum_i sum_j w_ij log P(output_ij | input_i).
        logps = sequence_logps(outputs.logits, inputs['labels'],
                              normalize=self.objective['length_normalize'])
        loss = -(logps * inputs['weights']).sum() / inputs['task_count']
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                        ignore_keys=None, metric_key_prefix='eval'):
        from transformers.trainer_utils import EvalLoopOutput
        # A batch contains tasks with variable reference counts. Aggregate by tasks,
        # not by the flattened sequence count inferred by Trainer's default loop.
        model = self.model_wrapped
        was_training = model.training
        total_loss, count = 0.0, 0
        model.eval()
        try:
            with torch.no_grad():
                for batch in dataloader:
                    batch = self._prepare_inputs(batch)
                    with self.compute_loss_context_manager():
                        loss = self.compute_loss(model, batch)
                    size = batch['task_count']
                    total_loss += loss.item() * size
                    count += size
                    self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)
        finally:
            model.train(was_training)
        if not count:
            raise ValueError('Validation dataset is empty')
        return EvalLoopOutput(predictions=None, label_ids=None,
                              metrics={metric_key_prefix + '_loss': total_loss / count}, num_samples=count)


def train_sft(args, model, tokenizer):
    stage = 'sft'
    objective = {'length_normalize': args.length_normalize}
    dataset = load_stage_dataset(args, tokenizer, stage)
    val_dataset = load_stage_dataset(args, tokenizer, stage, split='val')
    from pathlib import Path
    import json
    output = Path(args.output_dir) / stage
    resume = args.trainer_checkpoint
    if resume:
        output = Path(resume).resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    (output / 'training_args.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    # Restore configured dropout before an independently invoked stage.
    for name, module in model.named_modules():
        if 'lora_dropout' in name and isinstance(module, torch.nn.Dropout):
            module.p = args.lora_dropout
    options = dict(
        output_dir=str(output), num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps, logging_steps=args.logging_steps,
        log_level=args.log_level.lower(), logging_nan_inf_filter=False,
        eval_strategy=args.eval_strategy, eval_steps=args.eval_steps,
        prediction_loss_only=True,
        save_strategy='steps', save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.device == 'cuda', use_cpu=args.device == 'cpu',
        fp16=args.dtype == 'float16', bf16=args.dtype == 'bfloat16',
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        report_to='none', optim='adamw_torch',
        remove_unused_columns=False, label_names=['labels'],
    )
    trainer = SFTTrainer(model=model, args=TrainingArguments(**options),
                               train_dataset=dataset, eval_dataset=val_dataset,
                               data_collator=Collator(tokenizer),
                               callbacks=[make_training_log_callback(stage)],
                               stage=stage, objective=objective)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output / 'final'))
    tokenizer.save_pretrained(output / 'final')
    print(f'[{stage}] Final adapter and tokenizer saved: {output / "final"}')
    return output / 'final'

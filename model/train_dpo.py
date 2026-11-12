import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from tokenize_utils import Collator
from utils import sequence_logps
from data_loader import load_stage_dataset
from run_logging import make_training_log_callback


class DPOTrainer(Trainer):
    def __init__(self, *args, stage, objective, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage, self.objective = stage, objective
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logps = sequence_logps(outputs.logits, inputs['labels'])
        reference = inputs['reference_logps']
        # Each pair is flattened in chosen, rejected order.
        margin = (logps[0::2] - logps[1::2]) - (reference[0::2] - reference[1::2])
        loss = (-F.logsigmoid(self.objective['beta'] * margin) * inputs['weights'][0::2]).mean()
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


class ReferenceCollator(Collator):
    def __call__(self, rows):
        batch = super().__call__(rows)
        batch['reference_logps'] = torch.tensor([v for row in rows for v in row['reference_logps']])
        return batch


def train_dpo(args, model, tokenizer):
    stage = 'dpo'
    objective = {'beta': args.dpo_beta}
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
    if stage == 'dpo':
        # Cache the frozen current-SFT policy before any DPO updates.
        # Resuming must reuse the original cache, never recompute from updated weights.
        import hashlib
        import json
        for split, split_dataset in [('train', dataset), ('val', val_dataset)]:
            cache = output / ('reference_logps.pt' if split == 'train' else 'val_reference_logps.pt')
            print(f'[dpo/{split}] Reference cache: {cache}; pairs={len(split_dataset)}; resume={bool(resume)}')
            fingerprint = hashlib.sha256(json.dumps(split_dataset, sort_keys=True).encode()).hexdigest()
            if resume:
                saved = torch.load(cache, map_location='cpu', weights_only=True)
                if saved['fingerprint'] != fingerprint:
                    raise ValueError(f'DPO resume {split} dataset differs from original')
                values = saved['values']
            else:
                values = []
                was_training = model.training
                model.eval()
                collator = Collator(tokenizer)
                try:
                    with torch.no_grad():
                        for item in split_dataset:
                            batch = collator([item])
                            batch = {k: v.to(model.device) for k, v in batch.items() if k in ('input_ids', 'attention_mask', 'labels')}
                            values.append(sequence_logps(model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits, batch['labels']).cpu().tolist())
                finally:
                    model.train(was_training)
                torch.save({'fingerprint': fingerprint, 'values': values}, cache)
                print(f'[dpo/{split}] Reference cache saved: {cache}')
            if len(values) != len(split_dataset):
                raise ValueError(f'Invalid {split} reference cache length')
            for item, value in zip(split_dataset, values):
                item['reference_logps'] = value
        # DPO likelihoods use deterministic policies (including LoRA dropout).
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    options = dict(
        output_dir=str(output), num_train_epochs=args.dpo_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.dpo_learning_rate,
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
    trainer = DPOTrainer(model=model, args=TrainingArguments(**options),
                               train_dataset=dataset, eval_dataset=val_dataset,
                               data_collator=ReferenceCollator(tokenizer),
                               callbacks=[make_training_log_callback(stage)],
                               stage=stage, objective=objective)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output / 'final'))
    tokenizer.save_pretrained(output / 'final')
    print(f'[{stage}] Final adapter and tokenizer saved: {output / "final"}')
    return output / 'final'

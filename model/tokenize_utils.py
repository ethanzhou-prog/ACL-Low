class SequenceTooLongError(ValueError):
    pass


def prompt_text(row):
    return row['instruction'] + ('\n' + row['input'] if row.get('input') else '') + '\n'


def encode(tokenizer, prompt, output, max_length):
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    target = tokenizer.encode(output, add_special_tokens=False) + [tokenizer.eos_token_id]
    if not prefix or tokenizer.eos_token_id is None:
        raise ValueError('Prompt and EOS token are required')
    if len(prefix) + len(target) > max_length:
        raise SequenceTooLongError(
            f'Sequence has {len(prefix) + len(target)} tokens; max_seq_length={max_length}')
    return {'input_ids': prefix + target, 'labels': [-100] * len(prefix) + target}


class Collator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, rows):
        import torch
        sequences, weights = [], []
        for row in rows:
            sequences.extend(row['sequences'])
            weights.extend(row['weights'])
        width = max(len(s['input_ids']) for s in sequences)
        return {
            'input_ids': torch.tensor([s['input_ids'] + [self.pad_id] * (width - len(s['input_ids'])) for s in sequences]),
            'attention_mask': torch.tensor([[1] * len(s['input_ids']) + [0] * (width - len(s['input_ids'])) for s in sequences]),
            'labels': torch.tensor([s['labels'] + [-100] * (width - len(s['labels'])) for s in sequences]),
            'weights': torch.tensor(weights), 'task_count': len(rows),
        }

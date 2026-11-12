import torch.nn.functional as F


def sequence_logps(logits, labels, normalize=False):
    labels = labels[:, 1:]
    losses = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)).float(),
                             labels.contiguous().view(-1), ignore_index=-100, reduction='none')
    values = -losses.view_as(labels).sum(-1)
    return values / labels.ne(-100).sum(-1).clamp_min(1) if normalize else values

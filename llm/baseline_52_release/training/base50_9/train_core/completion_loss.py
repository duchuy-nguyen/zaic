"""Only the two actual V2 collation/loss helpers; no generic Kaggle trainer."""

def collate_rows(rows, pad_id, torch):
    max_len = max((len(r['input_ids']) for r in rows))
    return {'input_ids': torch.tensor([r['input_ids'] + [pad_id] * (max_len - len(r['input_ids'])) for r in rows]), 'attention_mask': torch.tensor([[1] * len(r['input_ids']) + [0] * (max_len - len(r['input_ids'])) for r in rows]), 'labels': torch.tensor([r['labels'] + [-100] * (max_len - len(r['labels'])) for r in rows])}

def completion_causal_loss(model, inputs, num_items_in_batch=None):
    """Same masked next-token CE, but project only supervised positions to vocabulary logits.

    Keep the complete input sequence/backbone computation. Position t predicts label t+1.
    Qwen3.5 supports a tensor of sequence indices through logits_to_keep.
    """
    import torch
    import torch.nn.functional as F
    labels = inputs['labels']
    shifted = labels[:, 1:]
    positions = torch.nonzero(shifted.ne(-100).any(dim=0), as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError('No supervised next-token targets; refuse zero-loss training')
    arguments = {k: v for k, v in inputs.items() if k not in ('labels', 'num_items_in_batch', 'use_cache')}
    outputs = model(**arguments, labels=None, use_cache=False, logits_to_keep=positions)
    logits = outputs.logits
    if logits.shape[:2] != (labels.shape[0], positions.numel()):
        raise ValueError('Model did not honor indexed logits_to_keep; stop rather than silently use full logits')
    targets = shifted.index_select(1, positions)
    numerator = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100, reduction='sum')
    denominator = targets.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
    denominator = torch.as_tensor(denominator, device=numerator.device, dtype=numerator.dtype)
    if denominator.numel() != 1 or denominator.item() <= 0:
        raise ValueError('Invalid supervised-token normalization')
    return (numerator / denominator, outputs)

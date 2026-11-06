"""Exact full-context attention with bounded query tiles and recomputed backward.

No sliding window, no token truncation, no approximate attention. In particular,
this avoids materializing an entire [batch, heads, sequence, sequence] tensor
when Turing GPUs cannot use the fused GQA path chosen by native SDPA.
"""
from dataclasses import dataclass


@dataclass
class DeferredMask:
    kwargs: dict


def deferred_mask(**kwargs):
    # Preserve HF's causal/cache/padding/packing semantics, but construct each
    # query tile inside the checkpoint instead of allocating an L x L mask.
    return DeferredMask(kwargs)


def attention_forward(module, query, key, value, attention_mask, dropout=0.0,
                      scaling=None, is_causal=None, query_chunk=128, **kwargs):
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from torch.utils.checkpoint import checkpoint
    if dropout:
        raise ValueError('Bounded attention requires attention dropout=0; no silent RNG change')
    if query_chunk < 1: raise ValueError('Positive attention query chunk required')
    groups = query.shape[1] // key.shape[1]
    if key.shape[1] * groups != query.shape[1]: raise ValueError('Invalid GQA head ratio')
    # Native enable_gqa can select the quadratic math fallback on T4. Repeat only
    # the linear-size K/V tensors explicitly; no GQA dispatch is needed below.
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    q_length, k_length = query.shape[-2], key.shape[-2]
    causal = getattr(module, 'is_causal', True) if is_causal is None else is_causal
    blocks = []
    for start in range(0, q_length, query_chunk):
        end = min(start + query_chunk, q_length)
        def block(q, k, v, start=start, end=end):
            if isinstance(attention_mask, DeferredMask):
                from transformers.masking_utils import sdpa_mask
                options = dict(attention_mask.kwargs)
                # Accelerate moves tensors across shards, but not this deferred
                # metadata object. Construct the tile on the current layer's GPU.
                options['device']=q.device
                for name,value in list(options.items()):
                    if isinstance(value,torch.Tensor):options[name]=value.to(q.device)
                options.update(q_length=end-start,
                    q_offset=options.get('q_offset', 0) + start,
                    allow_is_causal_skip=False, allow_is_bidirectional_skip=False)
                mask = sdpa_mask(**options)
            else:
                # Standalone path used by equivalence tests and non-cached text.
                q_index = torch.arange(start, end, device=q.device) + k_length - q_length
                k_index = torch.arange(k_length, device=q.device)
                mask = (k_index[None, :] <= q_index[:, None]) if causal else None
                if attention_mask is not None:
                    if attention_mask.dim() == 2:
                        padding = attention_mask[:, None, None, :k_length].bool()
                        mask = padding if mask is None else mask[None, None] & padding
                    elif attention_mask.dim() == 4:
                        custom = attention_mask[..., :k_length]
                        if custom.shape[-2] != 1: custom = custom[..., start:end, :]
                        if custom.dtype == torch.bool:
                            mask = custom if mask is None else mask & custom
                        else:
                            if mask is not None:
                                custom = custom.masked_fill(~mask, torch.finfo(q.dtype).min)
                            mask = custom
                    else:
                        raise ValueError('Unsupported attention mask rank')
            # MATH is portable to T4 and deterministic at dropout=0. Its temporary
            # score allocation is bounded by query_chunk, even without Flash kernels.
            with sdpa_kernel([SDPBackend.MATH]):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                    dropout_p=0., is_causal=False, scale=scaling, enable_gqa=False)
        q = query[..., start:end, :]
        if torch.is_grad_enabled() and any(t.requires_grad for t in (q, key, value)):
            # Without recomputation, autograd would retain every tile's softmax
            # tensors and the sum would still consume quadratic training memory.
            out = checkpoint(block, q, key, value, use_reentrant=False)
        else:
            out = block(q, key, value)
        blocks.append(out)
    output = torch.cat(blocks, dim=-2)
    return output.transpose(1, 2).contiguous(), None


def register(query_chunk=128):
    from functools import partial
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    name = 'lifemate_bounded_' + str(query_chunk)
    AttentionInterface.register(name, partial(attention_forward, query_chunk=query_chunk))
    AttentionMaskInterface.register(name, deferred_mask)
    return name


def completion_loss(model, inputs, target_chunk=32):
    """Compute exact masked causal CE without retaining all target x vocab logits."""
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    if target_chunk<1:raise ValueError('Positive target loss chunk required')
    labels = inputs['labels']
    positions = torch.where((labels[:, 1:] != -100).any(dim=0))[0]
    if not positions.numel(): raise ValueError('No supervised next-token positions')
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    output = base.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                        use_cache=False, return_dict=True)
    hidden = output[0] if isinstance(output, tuple) else output.last_hidden_state
    targets = labels[:, 1:].index_select(1, positions)
    total = None
    for start in range(0, positions.numel(), target_chunk):
        chosen = positions[start:start+target_chunk]
        h = hidden.index_select(1, chosen)
        y = targets[:, start:start+target_chunk]
        def loss_block(h, y):
            logits = base.lm_head(h)
            return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), ignore_index=-100, reduction='sum')
        part = checkpoint(loss_block, h, y, use_reentrant=False)
        total = part if total is None else total + part
    loss = total / (targets != -100).sum()
    if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
    return loss, None

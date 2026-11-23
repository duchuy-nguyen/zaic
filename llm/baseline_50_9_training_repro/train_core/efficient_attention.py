"""Explicit KV expansion + memory-efficient SDPA. No approximation, dtype change or truncation."""
def expand_grouped_kv(query, key, value, enable_gqa):
    if enable_gqa:
        qh, kh, vh = query.shape[-3], key.shape[-3], value.shape[-3]
        if kh != vh or qh % kh:
            raise ValueError('Invalid grouped-query attention head counts')
        key = key.repeat_interleave(qh//kh, dim=-3)
        value = value.repeat_interleave(qh//vh, dim=-3)
    return query, key, value


def efficient_sdpa(original, query, key, value, attn_mask=None, dropout_p=0.0,
                   is_causal=False, scale=None, enable_gqa=False):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    query,key,value=expand_grouped_kv(query,key,value,enable_gqa)
    if attn_mask is not None:
        # SDPA bool semantics: True attends, False is -inf additive bias.
        if attn_mask.dtype == torch.bool:
            attn_mask=torch.zeros_like(attn_mask,dtype=query.dtype).masked_fill(~attn_mask,float('-inf'))
        # CUTLASS bias row strides must be aligned; pad storage only, keep the exact logical mask/sequence.
        if attn_mask.ndim>=2 and (attn_mask.stride(-2)%8 or attn_mask.stride(-1)!=1):
            width=attn_mask.shape[-1]
            padded=attn_mask.new_empty((*attn_mask.shape[:-1],(width+7)//8*8))
            padded[...,:width].copy_(attn_mask)
            attn_mask=padded[...,:width]
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return original(query,key,value,attn_mask=attn_mask,dropout_p=dropout_p,
                        is_causal=is_causal,scale=scale,enable_gqa=False)


def probe():
    """Small forward/backward checks on the actual GPU; never load a language model."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend,sdpa_kernel
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for attention-kernel probe')
    torch.manual_seed(20261103)
    original=F.scaled_dot_product_attention
    cases=[]
    for dtype in [torch.float32,torch.float16]:
        for masked in [False,True]:
            n=65 if masked else 64
            xs=[torch.randn(1,h,n,256,device='cuda',dtype=dtype,requires_grad=True) for h in [16,4,4]]
            mask=None
            if masked:
                mask=torch.ones(n,n,device='cuda',dtype=torch.bool).tril().view(1,1,n,n)
                mask=mask & (torch.arange(n,device='cuda')<n-1).view(1,1,1,n)
            with sdpa_kernel(SDPBackend.MATH):
                reference=original(*xs,attn_mask=mask,is_causal=not masked,enable_gqa=True)
            expected=torch.autograd.grad(reference.float().square().mean(),xs)
            actual=efficient_sdpa(original,*xs,attn_mask=mask,is_causal=not masked,enable_gqa=True)
            gradients=torch.autograd.grad(actual.float().square().mean(),xs)
            tolerance=dict(rtol=2e-2,atol=2e-3) if dtype==torch.float16 else dict(rtol=5e-4,atol=5e-5)
            torch.testing.assert_close(actual,reference,**tolerance)
            for a,b in zip(gradients,expected):torch.testing.assert_close(a,b,**tolerance)
            cases.append({'dtype':str(dtype),'masked':masked,'length':n,'max_output_error':(actual-reference).abs().max().item()})
    torch.cuda.synchronize()
    report={'gpu':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability()),
            'torch':torch.__version__,'efficient_forward_backward_passed':True,'cases':cases}
    print('EFFICIENT_ATTENTION_PROBE_PASSED',report,flush=True)
    return report


def install():
    import torch.nn.functional as F
    if getattr(F.scaled_dot_product_attention,'_lifemate_efficient_only',False):return
    original=F.scaled_dot_product_attention
    def wrapped(query,key,value,attn_mask=None,dropout_p=0.0,is_causal=False,*,scale=None,enable_gqa=False):
        if query.device.type!='cuda':
            return original(query,key,value,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal,
                            scale=scale,enable_gqa=enable_gqa)
        return efficient_sdpa(original,query,key,value,attn_mask,dropout_p,is_causal,scale,enable_gqa)
    wrapped._lifemate_efficient_only=True
    F.scaled_dot_product_attention=wrapped
    print('Attention policy: explicit equivalent KV expansion, efficient SDPA only; math fallback disabled.',flush=True)


if __name__=='__main__':probe()

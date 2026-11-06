"""Numerical forward/backward checks. No model downloads or BTC data."""
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bounded_attention


class BoundedAttentionTests(unittest.TestCase):
    def test_deferred_mask_uses_current_layer_device_not_embedding_device(self):
        import torch
        def make_mask(**kwargs):
            self.assertEqual(str(kwargs['device']),'cpu')
            self.assertEqual(kwargs['attention_mask'].device.type,'cpu')
            return torch.arange(kwargs['kv_length'])[None,:] <= (
                torch.arange(kwargs['q_length'])[:,None]+kwargs['q_offset'])
        metadata=bounded_attention.DeferredMask(dict(device='cuda:99',q_length=5,kv_length=5,
            q_offset=0,attention_mask=torch.ones(1,5)))
        q=torch.randn(1,2,5,4,requires_grad=True)
        with patch.dict(sys.modules,{'transformers.masking_utils':SimpleNamespace(sdpa_mask=make_mask)}):
            output,_=bounded_attention.attention_forward(SimpleNamespace(is_causal=True),q,q,q,metadata,query_chunk=2)
            output.sum().backward()

    def test_completion_logits_remain_bounded_during_backward(self):
        import torch
        class Backbone(torch.nn.Module):
            def __init__(self):super().__init__();self.embed=torch.nn.Embedding(20,8)
            def forward(self,input_ids,**kwargs):return SimpleNamespace(last_hidden_state=self.embed(input_ids))
        class Head(torch.nn.Linear):
            observed=[]
            def forward(self,x):
                self.observed.append(x.shape[1]);self_size=x.shape[1]
                if self_size>2:raise AssertionError('Unbounded completion logits')
                return super().forward(x)
        model=SimpleNamespace(model=Backbone(),lm_head=Head(8,4096))
        inputs={'input_ids':torch.tensor([[1,2,3,4,5,6,7,8]]),
                'attention_mask':torch.ones(1,8,dtype=torch.long),
                'labels':torch.tensor([[-100,2,3,4,5,6,7,8]])}
        loss,_=bounded_attention.completion_loss(model,inputs,target_chunk=2)
        before=len(model.lm_head.observed);loss.backward()
        self.assertGreater(len(model.lm_head.observed),before)
        self.assertTrue(all(n<=2 for n in model.lm_head.observed))

    def test_exact_outputs_and_gradients_for_gqa_padding_and_cached_queries(self):
        import torch
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        torch.manual_seed(9)
        for qlen, klen, padded in ((7,7,False), (7,7,True), (3,9,False)):
            with self.subTest(qlen=qlen, klen=klen, padded=padded):
                tensors = [torch.randn(2,4,qlen,4,dtype=torch.float64,requires_grad=True),
                           torch.randn(2,2,klen,4,dtype=torch.float64,requires_grad=True),
                           torch.randn(2,2,klen,4,dtype=torch.float64,requires_grad=True)]
                original = [t.detach().clone().requires_grad_() for t in tensors]
                padding = torch.ones(2,klen,dtype=torch.long) if padded else None
                if padded: padding[0,:2] = 0
                mask = torch.arange(klen)[None,:] <= (torch.arange(qlen)[:,None]+klen-qlen)
                if padded: mask = mask[None,None] & padding[:,None,None,:].bool()
                expected = sdpa(original[0], original[1].repeat_interleave(2,dim=1),
                    original[2].repeat_interleave(2,dim=1), attn_mask=mask, scale=.3).transpose(1,2)
                observed,_ = bounded_attention.attention_forward(SimpleNamespace(is_causal=True),
                    *tensors, padding, scaling=.3, query_chunk=2)
                torch.testing.assert_close(observed,expected,atol=1e-10,rtol=1e-10)
                observed.square().sum().backward(); expected.square().sum().backward()
                for a,b in zip(tensors,original):
                    torch.testing.assert_close(a.grad,b.grad,atol=1e-9,rtol=1e-9)

    def test_scores_are_bounded_in_forward_and_recomputed_backward(self):
        import torch
        import torch.nn.functional as F
        original = F.scaled_dot_product_attention
        seen = []
        def observe(q,k,v,**kwargs):
            seen.append(q.shape[-2])
            self.assertLessEqual(q.shape[-2], 3)
            return original(q,k,v,**kwargs)
        q = torch.randn(1,4,11,8,requires_grad=True)
        k = torch.randn(1,2,11,8,requires_grad=True); v = torch.randn_like(k,requires_grad=True)
        with patch.object(F, 'scaled_dot_product_attention', side_effect=observe):
            out,_ = bounded_attention.attention_forward(SimpleNamespace(is_causal=True),q,k,v,None,query_chunk=3)
            forward_calls = len(seen)
            out.sum().backward()
        self.assertEqual(forward_calls,4)
        self.assertGreater(len(seen),forward_calls)


if __name__ == '__main__': unittest.main()

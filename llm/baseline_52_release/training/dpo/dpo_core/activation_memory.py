"""Offload large floating batched activations, never clone2D weights into host RAM."""
from contextlib import contextmanager

@contextmanager
def saved_activation_context():
    import torch
    def pack(tensor):
        if tensor.device.type=='cuda' and tensor.is_floating_point() and tensor.ndim>=3 and tensor.numel()>=65536:
            return (tensor.device,tensor.detach().to('cpu',non_blocking=False))
        return tensor
    def unpack(value):
        if isinstance(value,tuple):return value[1].to(value[0],non_blocking=False)
        return value
    with torch.autograd.graph.saved_tensors_hooks(pack,unpack):yield

def parity_test():
    import copy,torch
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(52)
    a=torch.nn.Sequential(torch.nn.Linear(32,64),torch.nn.SiLU(),torch.nn.Linear(64,32)).cuda();b=copy.deepcopy(a)
    x=torch.randn(1,2300,32,device='cuda',requires_grad=True);y=x.detach().clone().requires_grad_()
    aa=checkpoint(a,x,use_reentrant=False);aa.square().mean().backward()
    with saved_activation_context():bb=checkpoint(b,y,use_reentrant=False);bb.square().mean().backward()
    torch.testing.assert_close(aa,bb,rtol=1e-5,atol=1e-6);torch.testing.assert_close(x.grad,y.grad,rtol=1e-5,atol=1e-6)
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)
    return {'passed':True,'exact_saved_values':'CPU copy restored on CUDA','matching_outputs_and_gradients':True}
if __name__=='__main__':print(parity_test())

"""Exact torch.AdamW, FP32 moments on host outside optimizer.step()."""
import torch
class OffloadedAdamW(torch.optim.AdamW):
    def _host_states(self):
        for group in self.param_groups:
            for p in group['params']:
                for key,value in list(self.state[p].items()):
                    if key!='step' and isinstance(value,torch.Tensor):self.state[p][key]=value.to('cpu',non_blocking=False)
    @torch.no_grad()
    def step(self,closure=None):
        for group in self.param_groups:
            for p in group['params']:
                for key,value in list(self.state[p].items()):
                    if key!='step' and isinstance(value,torch.Tensor):self.state[p][key]=value.to(p.device,non_blocking=False)
        result=super().step(closure)
        self._host_states()
        return result
    def load_state_dict(self,state_dict):
        super().load_state_dict(state_dict);self._host_states()

def parity_test():
    import copy,io
    torch.manual_seed(66)
    a=torch.nn.Linear(32,19).cuda();b=copy.deepcopy(a)
    oa=torch.optim.AdamW(a.parameters(),lr=2e-5,weight_decay=.01)
    ob=OffloadedAdamW(b.parameters(),lr=2e-5,weight_decay=.01)
    for step in range(4):
        x=torch.randn(1,91,32,device='cuda');a(x).square().mean().backward();b(x).square().mean().backward()
        oa.step();ob.step();oa.zero_grad();ob.zero_grad()
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
        assert all(v.device.type=='cpu' for st in ob.state.values() for k,v in st.items() if isinstance(v,torch.Tensor))
        if step==1:
            f=io.BytesIO();torch.save(ob.state_dict(),f);f.seek(0);ob.load_state_dict(torch.load(f,weights_only=True))
    return {'passed':True,'bitwise_equal_to_torch_AdamW':True,'FP32_moments_preserved':True,'checkpoint_roundtrip':True}
if __name__=='__main__':print(parity_test())

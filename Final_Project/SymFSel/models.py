# models.py
import torch
from torch import nn

# You might import PTABLE from config.py or define it here if more self-contained
# from config import PTABLE # Example

class SpeciesMLP(nn.Module):
    def __init__(self,n_g,hid,depth):
        super().__init__()
        layers=[nn.Linear(n_g,hid), nn.GELU()]
        for _ in range(depth-1): layers += [nn.Linear(hid,hid), nn.GELU()]
        layers.append(nn.Linear(hid,1))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class AtomicMLP(nn.Module):
    def __init__(self,n_g,n_sp,hid,depth):
        super().__init__()
        self.f=nn.ModuleList(SpeciesMLP(n_g,hid,depth) for _ in range(n_sp))
    def forward(self,sym,g,blk,n_blk):
        model_param_dtype = self.f[0].net[0].weight.dtype
        e=torch.empty(sym.size(0),dtype=model_param_dtype, device=g.device)
        for s_val in torch.unique(sym):
            m=(sym==s_val)
            if m.any():
                e[m]=self.f[s_val](g[m]).squeeze(-1).to(e.dtype)
        tot=torch.zeros(n_blk,dtype=e.dtype,device=g.device)
        tot.scatter_add_(0,blk,e) 
        return tot

class AtomicLin(nn.Module):
    def __init__(self,n_g,n_sp):
        super().__init__()
        self.f=nn.ModuleList(nn.Linear(n_g,1) for _ in range(n_sp))
    def forward(self,sym,g,blk,n_blk):
        model_param_dtype = self.f[0].weight.dtype
        e=torch.empty(sym.size(0),dtype=model_param_dtype, device=g.device)
        for s_val in torch.unique(sym):
            m=(sym==s_val)
            if m.any():
                e[m]=self.f[s_val](g[m]).squeeze(-1).to(e.dtype)
        tot=torch.zeros(n_blk,dtype=e.dtype,device=g.device)
        tot.scatter_add_(0,blk,e)
        return tot
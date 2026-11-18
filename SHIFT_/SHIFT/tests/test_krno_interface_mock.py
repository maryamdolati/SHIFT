import sys
import types
import torch
from torch import nn


def test_original_bridge_grid_shapes_via_fake_api(monkeypatch):
    from shift_krno.model import SHIFT
    pkg=types.ModuleType('khatriraonop')
    def quad(v):
        return (v,torch.ones_like(v)/len(v))
    class FauxOperator(nn.Module):
        def __init__(self,out):
            super().__init__();self.out=out
            self.scalar=nn.Parameter(torch.tensor(1.))
        def super_resolution(self,out_grid,in_grid,z,latent_grid=None):
            assert latent_grid is None
            return torch.ones(z.shape[0],len(out_grid[0][0]),self.out,device=z.device)*self.scalar
    class ModelClass:
        @staticmethod
        def easy_init(d,**kwargs):
            assert d==1
            return FauxOperator(kwargs['out_channels'])
    pkg.models=types.SimpleNamespace(KhatriRaoNO_v2=ModelClass)
    pkg.quadrature=types.SimpleNamespace(trapezoidal_vecs_uneven=quad)
    monkeypatch.setitem(sys.modules,'khatriraonop',pkg)
    mdl=SHIFT(2,hidden=12,backbone='original')
    x=torch.randn(2,10,2);mask=torch.ones_like(x)
    t=torch.arange(10).float()[None].expand(2,-1)
    q=torch.arange(10,15).float()[None].expand(2,-1)
    out=mdl(x,mask,t,q,K=2)
    assert out.mean.shape==(2,5,2)
    assert out.component_mean.shape==(2,2,5,2)

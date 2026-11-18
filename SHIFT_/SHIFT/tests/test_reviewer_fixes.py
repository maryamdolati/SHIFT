from contextlib import contextmanager
import sys
import types
import numpy as np
import pytest
import torch
from torch import nn


@contextmanager
def fake_krno(monkeypatch):
    pkg=types.ModuleType('khatriraonop')
    captures=[]

    def quad(x):
        return (torch.ones_like(x)/len(x), x)

    def quadrule(n,a,b):
        return (torch.ones(n)*((b-a)/n),torch.linspace(a,b,n))

    def grid(fns,counts,lows,highs):
        rules=[fn(n,a,b) for fn,n,a,b in zip(fns,counts,lows,highs)]
        return ([r[0] for r in rules],[r[1] for r in rules])

    class Stub(nn.Module):
        def __init__(self,dim,out):
            super().__init__()
            self.dim=dim;self.out=out
            self.scale=nn.Parameter(torch.tensor(1.0))
        def super_resolution(self,out_grid,in_grid,z,latent_grid=None):
            assert latent_grid is None

            assert out_grid[0][0].numel()==out_grid[1][0].numel()
            captures.append((self.dim, [v.clone() for v in out_grid[1]],
                             [v.clone() for v in in_grid[1]]))
            if self.dim==1:
                value=z.mean(dim=1,keepdim=True).mean(dim=-1,keepdim=True)
                return value.expand(z.size(0),out_grid[1][0].numel(),self.out)*self.scale
            raise AssertionError('Spatial task must use official KhatriRaoNO forward')
        def forward(self, cart_grid, z):
            assert self.dim == 3
            assert cart_grid[0][0].numel() == 5
            captures.append((3, [q.clone() for q in cart_grid[1]],
                             [q.clone() for q in cart_grid[1]]))
            value=z.mean(dim=(1,2,3),keepdim=True).mean(dim=-1,keepdim=True)
            return value.expand(z.size(0),5,32,32,self.out)*self.scale

    class Factory:
        @staticmethod
        def easy_init(dim,**kwargs):
            assert kwargs['lifting_channels']==128
            assert kwargs['n_integral_layers']==3
            return Stub(dim,kwargs['out_channels'])
    pkg.models=types.SimpleNamespace(KhatriRaoNO_v2=Factory,KhatriRaoNO=Factory)
    def cart_grid(g):
        weights, nodes = g
        return (weights, [torch.cartesian_prod(x, x) for x in nodes])
    pkg.quadrature=types.SimpleNamespace(trapezoidal_vecs_uneven=quad,
        midpoint_vecs=quadrule,trapezoidal_vecs=quadrule,get_quad_grid=grid,
        quad_to_cartesian_grid=cart_grid)
    monkeypatch.setitem(sys.modules,'khatriraonop',pkg)
    yield captures


def test_query_nodes_are_not_quadrature_weights(monkeypatch):
    from shift_krno.model import OriginalKRNOBridge
    with fake_krno(monkeypatch) as calls:
        net=OriginalKRNOBridge(2,2,predictive_variance=False)
        x=torch.randn(1,4,2)
        q=torch.tensor([[2.5]])
        pred=net(x,q)
        assert pred.shape==(1,1,2)
        torch.testing.assert_close(calls[-1][1][0],q[0])

        inp=torch.tensor([[0.,0.1,.6,1.],[0.,.3,.7,1.]])
        net(x.expand(2,-1,-1),q.expand(2,-1),input_coords=inp)
        assert len(calls)==3
        torch.testing.assert_close(calls[-2][2][0],inp[0])
        torch.testing.assert_close(calls[-1][2][0],inp[1])


def test_reference_and_warp_fixed_same_operator_path(monkeypatch):
    from shift_krno.model import KRNOReference,KRNOWarpOnly
    with fake_krno(monkeypatch):
        ref=KRNOReference(2)
        warp=KRNOWarpOnly(2)
        warp.backbone.load_state_dict(ref.backbone.state_dict())
        x=torch.randn(2,8,2)
        mask=torch.ones_like(x)
        t=torch.tensor([[0.,.1,.3,.6,.7,.8,.9,1.],
                        [0.,.2,.25,.5,.7,.75,.88,1.]])
        q=torch.tensor([[1.1,1.5],[1.2,1.4]])
        a=ref(x,mask,t,q,K=3)
        b=warp(x,mask,t,q,K=3,mode='fixed')
        torch.testing.assert_close(a.mean,b.mean)
        torch.testing.assert_close(a.u_temporal,torch.zeros_like(a.u_temporal))
        assert not hasattr(warp,'multiscale')
        assert not hasattr(warp,'pixel_encoder')
        assert not hasattr(warp,'spectral')


def test_shuffled_clock_uses_actual_donor_map(monkeypatch):
    from shift_krno.model import ConditionalWarp
    x=torch.stack((torch.zeros(7,2),torch.ones(7,2)))[None] if False else torch.randn(3,7,2)
    mask=torch.ones_like(x)
    t=torch.arange(7.).expand(3,-1)
    net=ConditionalWarp(2)
    noise=torch.zeros(2,3,6)
    _,own,_,_,_,_=net(x,mask,t,K=2,mode='conditional',external_noise=noise)
    _,cross,_,_,_,_=net(x,mask,t,K=2,mode='shuffled',external_noise=noise)
    torch.testing.assert_close(cross,own.roll(1,1))


def test_spatial_model_same_krno_grid_and_60_total(monkeypatch):
    from shift_krno.model import SpatialKRNOReference,SpatialKRNOWarpOnly
    with fake_krno(monkeypatch) as calls:
        ref=SpatialKRNOReference(width=20)
        warp=SpatialKRNOWarpOnly(width=20)
        warp.backbone.load_state_dict(ref.backbone.state_dict())
        x=torch.ones(1,5,32,32,3)
        t=torch.arange(5.)[None]*.01
        q=torch.arange(5.,10.)[None]*.01
        a=ref(x,torch.ones_like(x),t,q,K=2)
        b=warp(x,torch.ones_like(x),t,q,K=2,mode='fixed')
        torch.testing.assert_close(a.mean,b.mean)
        assert a.mean.shape==(1,5,32,32,3)

        torch.testing.assert_close(calls[-1][2][1][0],torch.tensor([-1.,-1.]))
        torch.testing.assert_close(calls[-1][2][1][-1],torch.tensor([1.,1.]))


def test_m4_native_and_paper_validation_fraction(tmp_path):
    from shift_krno.data import load_m4
    import pandas as pd
    (tmp_path/'Train').mkdir();(tmp_path/'Test').mkdir()
    tr=pd.DataFrame([['A']+list(range(100))])
    te=pd.DataFrame([['A']+list(range(8))])
    tr.to_csv(tmp_path/'Train/Quarterly-train.csv',index=False,header=['V1']+[f'V{i}' for i in range(2,102)])
    te.to_csv(tmp_path/'Test/Quarterly-test.csv',index=False,header=['V1']+[f'V{i}' for i in range(2,10)])
    q=load_m4(tmp_path,'Quarterly',lookback=14,pred=8)
    assert q.metadata['validation_fraction']==.05
    assert load_m4(tmp_path,'Quarterly',lookback=14,pred=8,val_fraction=.1).metadata['validation_fraction']==.1


def test_pair_must_use_same_trained_variance_head():
    from shift_krno.run import parser
    a=parser().parse_args(['--dataset','mujoco','--data-root','x'])
    assert a.architecture=='krno_warp'
    assert a.objective=='mse'
    assert not a.probabilistic
    assert a.reg_strength==0


def test_irregular_mask_feature_contract_matches_in_both_arms(monkeypatch):

    from shift_krno.model import KRNOReference, KRNOWarpOnly
    with fake_krno(monkeypatch):
        ref=KRNOReference(2,mask_features=True)
        warp=KRNOWarpOnly(2,mask_features=True)
        warp.backbone.load_state_dict(ref.backbone.state_dict())
        x=torch.tensor([[[1.,2.],[3.,4.],[5.,6.],[7.,8.]],
                        [[2.,1.],[4.,3.],[6.,5.],[8.,7.]]])
        mask=torch.tensor([[[1.,0.],[1.,1.],[0.,1.],[1.,1.]],
                           [[0.,1.],[1.,1.],[1.,0.],[1.,1.]]])
        t=torch.tensor([[0.,.2,.5,1.],[0.,.3,.8,1.]])
        q=torch.tensor([[1.2,1.4],[1.3,1.5]])
        a=ref(x,mask,t,q,K=2)
        b=warp(x,mask,t,q,K=2,mode='fixed')
        torch.testing.assert_close(a.mean,b.mean)
        assert ref.backbone.operator.dim==1
        assert ref.backbone.operator.out==2

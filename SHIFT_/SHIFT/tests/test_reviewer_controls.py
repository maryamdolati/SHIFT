import importlib.util
from pathlib import Path
import torch
import pytest
from shift_krno.model import ConditionalWarp,transfer_donor_map
from shift_krno.losses import mixture_nll
from types import SimpleNamespace


def test_donor_transfer_different_lengths():
    dt=torch.tensor([[0.,.2,1.]])
    rt=torch.tensor([[0.,.1,.5,.8,1.]])
    phi=torch.tensor([[[0.,.7,1.]],[[0.,.3,1.]]])
    got=transfer_donor_map(phi,dt,rt)
    assert got.shape==(2,1,5)
    assert torch.all(got.diff(dim=-1)>0)
    torch.testing.assert_close(got[:,:,0],torch.zeros(2,1))
    torch.testing.assert_close(got[:,:,-1],torch.ones(2,1))
    torch.testing.assert_close(got[:,0,1],torch.tensor([.35,.15]))


def test_conditional_donor_b1():
    w=ConditionalWarp(2);x=torch.randn(1,5,2);t=torch.arange(5.)[None];m=torch.ones_like(x)
    w.donor=(torch.randn(1,3,2),torch.ones(1,3,2),torch.arange(3.)[None])
    z,phi,*_=w(x,m,t,K=3,mode='shuffled')
    assert phi.shape==(3,1,5) and z.shape[1]==5
    assert (phi.diff(dim=-1)>0).all()


def test_mixture_loss_trains_variance():
    m=torch.zeros(2,1,3,1,requires_grad=True);v=torch.ones_like(m,requires_grad=True)
    loss=mixture_nll(SimpleNamespace(component_mean=m,component_variance=v),torch.full((1,3,1),2.))
    loss.backward();assert v.grad.abs().sum()>0


def test_reviewer_plan_locked_controls(tmp_path):
    p=Path(__file__).parents[1]/'scripts/reviewer_studies.py';spec=importlib.util.spec_from_file_location('reviewer_studies',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    c=dict(epochs=1,patience=1,batch_size=1,grad_accum=1,lr=.001,weight_decay=0,width=8,kernel_hidden=8,train_samples=2,eval_samples=2,input_noise=.1,dropout=.1,objective='mix_nll',capacity_widths=[4,8,16])
    plan=mod.create_plan('mimic',tmp_path,tmp_path,tmp_path/'runs',c)
    assert len(plan['commands'])==45 and not plan['capacity_matching_verified']
    assert {row['arm'] for row in plan['commands']}==set(mod.ARMS)
    for row in plan['commands']:
        cmd=row['command'];assert '--test-forbidden' in cmd and cmd[cmd.index('--objective')+1]=='mix_nll'
    with pytest.raises(FileNotFoundError):mod.create_plan('mimic',tmp_path,tmp_path,tmp_path/'runs',c,test=True)
    c['train_samples']=1
    with pytest.raises(ValueError,match='K>=2'):mod.create_plan('mimic',tmp_path,tmp_path,tmp_path/'runs',c)


def test_ensemble_variance_is_between_member():
    from shift_krno.ensemble import FixedCoordinateEnsemble
    from shift_krno.model import ShiftOutput
    class Member(torch.nn.Module):
        has_predictive_variance=True
        def __init__(self,value):super().__init__();self.value=value
        def forward(self,x,mask,t,q,K=1,mode='fixed'):
            m=torch.full((x.size(0),q.size(1),1),self.value);v=torch.ones_like(m);z=torch.zeros(1)
            return ShiftOutput(m,v,v,torch.zeros_like(v),m[None],v[None],z,z,z,z,z,z)
    model=FixedCoordinateEnsemble([Member(1.),Member(3.)]);x=torch.ones(1,4,1);t=torch.arange(4.)[None];q=torch.arange(4.,6.)[None]
    out=model(x,x,t,q)
    assert out.mean.unique().item()==2. and out.u_value.unique().item()==1. and out.u_temporal.unique().item()==1. and out.variance.unique().item()==2.

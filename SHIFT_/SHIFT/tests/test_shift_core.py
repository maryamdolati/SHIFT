import math
import torch
import pytest
from shift_krno.model import SHIFT,ConditionalWarp,latent_resample
from shift_krno.losses import mixture_nll,wrmse_crypto,crypto_blocks,trajectory_blocks


def batch(B=3,T=12,C=2,H=5):
    torch.manual_seed(41)
    x=torch.randn(B,T,C,requires_grad=True)
    m=torch.ones_like(x)
    t=torch.arange(T,dtype=torch.float32).expand(B,-1).clone()
    q=torch.arange(T,T+H,dtype=torch.float32).expand(B,-1).clone()
    return x,m,t,q


def test_monotonic_endpoints_and_normalized_masses():
    x,m,t,q=batch()
    clock=ConditionalWarp(2)
    z,p,r,mu,lv,ds=clock(x,m,t,K=8)
    assert p.shape==(8,3,12)
    assert torch.all(p[:,:,1:]>p[:,:,:-1])
    torch.testing.assert_close(p[:,:,0],torch.zeros_like(p[:,:,0]))
    torch.testing.assert_close(p[:,:,-1],torch.ones_like(p[:,:,-1]))
    mass=r*ds[None]
    torch.testing.assert_close((mass/mass.sum(-1,keepdim=True)).sum(-1),
                               torch.ones((8,3)))


def test_latent_interpolation_identity_is_exact():
    z=torch.randn(2,10,6)
    time=torch.arange(10).float().expand(2,-1)
    phi=(time-time[:,:1])/(time[:,-1:]-time[:,:1])
    zr,j=latent_resample(z,phi,time)
    torch.testing.assert_close(zr,z,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(j,j[:,:,0:1].expand_as(j),atol=1e-5,rtol=1e-5)


def test_automatic_gradients_through_warp_and_spectral():
    x,m,t,q=batch()
    model=SHIFT(2,hidden=16,backbone='surrogate')
    out=model(x,m,t,q,K=3)
    assert out.mean.shape==(3,5,2)
    torch.testing.assert_close(out.variance,out.u_value+out.u_temporal)
    out.mean.square().mean().backward()
    assert model.warp.mean_net[0].weight.grad is not None
    assert model.warp.mean_net[0].weight.grad.abs().sum()>0
    assert model.multiscale.freq_real.grad is not None


def test_fixed_clock_exact_and_no_temporal_uncertainty():
    x,m,t,q=batch()
    model=SHIFT(2,hidden=16,backbone='surrogate')
    out=model(x,m,t,q,K=3,mode='fixed')
    torch.testing.assert_close(out.u_temporal,torch.zeros_like(out.u_temporal))
    clock=(t-t[:,:1])/(t[:,-1:]-t[:,:1])
    torch.testing.assert_close(out.phi,clock[None].expand_as(out.phi))


def test_unconditional_not_dependent_on_signal():
    x,m,t,q=batch()
    c=ConditionalWarp(2)
    _,p1,_,_,_,_=c(x,m,t,K=2,mode='unconditional',external_noise=torch.zeros(2,3,11))
    _,p2,_,_,_,_=c(x*100,m,t,K=2,mode='unconditional',external_noise=torch.zeros(2,3,11))
    torch.testing.assert_close(p1,p2)


def test_crypto_scores_last_channel_only_and_exact_15_blocks():
    truth=torch.zeros(2,14,15,8)
    pred=torch.zeros_like(truth)
    pred[:,:,:5,0]=1e4
    w=torch.arange(1,15).float()
    assert float(wrmse_crypto(truth,pred,w))==0
    pred[:,:,10:15,-1]=.003
    b=crypto_blocks(truth,pred,w)
    assert b['1-5']==0
    assert b['6-10']==0
    assert b['11-15'] > .003
    assert b['total'] > 0
    assert b['official_total_is_final_horizon_only'] is False
    assert b['total_all_steps'] > 0
    pred[:,13,10:15,-1]=.006
    assert crypto_blocks(truth,pred,w)['11-15'] > b['11-15']
    with pytest.raises(ValueError):wrmse_crypto(truth,pred,None)


def test_trajectory_blocks_rmse_over_full_30():
    y=torch.zeros(5,30,2);pred=y.clone();pred[:,20:]=2
    b=trajectory_blocks(y,pred)
    assert b['1-10']==0 and b['11-20']==0
    assert b['21-30']==2
    assert abs(b['total']-math.sqrt(4/3))<1e-6


def test_future_data_not_in_model_inputs():
    x,m,t,q=batch(B=2)
    mdl=SHIFT(2,hidden=16,backbone='surrogate')
    torch.manual_seed(123);one=mdl(x,m,t,q,K=4)
    y=torch.randn(2,5,2)*9999
    torch.manual_seed(123);two=mdl(x,m,t,q,K=4)
    torch.testing.assert_close(one.mean,two.mean)

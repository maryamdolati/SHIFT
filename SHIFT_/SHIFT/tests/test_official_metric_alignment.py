from types import SimpleNamespace
import numpy as np
import pytest
import torch
from shift_krno.losses import wrmse_crypto, crypto_blocks, shallow_relative_l2
from shift_krno.run import validate


def test_exact_original_knf_wrmses_and_horizon_indexing():
    rng = np.random.default_rng(17)
    truth = rng.normal(size=(3, 14, 15, 8)).astype(np.float32)
    pred = truth + rng.normal(scale=.01, size=truth.shape).astype(np.float32)
    weights = np.array([4.30406509320417,6.779921907472252,2.3978952727983707,
        4.406719247264253,3.555348061489413,1.3862943611198906,
        5.8944028342648505,2.079441541679836,1.0986122886681098,
        2.3978952727983707,1.0986122886681098,1.6094379124341005,
        2.079441541679836,1.791759469228055], dtype=np.float32)

    raw = ((pred-truth) * weights[None,:,None,None])**2
    assert float(wrmse_crypto(torch.tensor(truth),torch.tensor(pred),torch.tensor(weights))) == pytest.approx(
        np.sqrt(np.mean(raw[...,-1])),rel=1e-6)
    scores=crypto_blocks(torch.tensor(truth),torch.tensor(pred),torch.tensor(weights))
    assert scores['official_short'] == pytest.approx(np.sqrt(raw[...,:5,-1].mean()),rel=1e-6)
    assert scores['official_medium'] == pytest.approx(np.sqrt(raw[...,5:10,-1].mean()),rel=1e-6)
    assert scores['official_long_step'] == pytest.approx(np.sqrt(raw[...,-15//3,-1].mean()),rel=1e-6)
    assert scores['total'] == pytest.approx(np.sqrt(raw[...,-1].mean()),rel=1e-6)
    assert scores['total_all_steps'] == pytest.approx(np.sqrt(raw[...,-1].mean()),rel=1e-6)


def test_crypto_validation_checkpoint_metric_is_quadrature_mse():
    class Zero(torch.nn.Module):
        def forward(self,x,mask,t,q,K=2,mode='fixed'):
            return SimpleNamespace(mean=torch.zeros((x.size(0),q.size(1),x.size(-1))))
    b=14
    mask=torch.ones(b,4,8)
    y=torch.full((b,15,8),300.)
    y[:,-1,-1]=1.0
    std=torch.arange(1,15).float()
    weights=torch.arange(1,15).float()
    a=SimpleNamespace(dataset='crypto',eval_samples=1,input_noise=0.,dropout=0.,mode='fixed',
        _metadata=dict(lookback=4,weights=weights.tolist(),per_asset_std=torch.stack([torch.ones(8)*s for s in std]).tolist()))
    batch=dict(x=torch.zeros(b,4,8),mask=mask,t=torch.arange(4.).expand(b,-1),
        q=torch.arange(4.,19.).expand(b,-1),y=y,ymask=torch.ones_like(y),series=torch.arange(b))
    score=validate(Zero(),[batch],a,torch.device('cpu'))
    expected=(y.square().mean()*15/4).sqrt()
    assert score == pytest.approx(expected.item(),rel=1e-6)


def test_shallow_denominator_excludes_observed_history():
    target=torch.ones(2,60,32,32,3)
    target[:,:5]=10000.
    pred=target.clone(); pred[:,5:]=0.

    wrong=shallow_relative_l2(target,pred)
    corrected=shallow_relative_l2(target,pred,observed_steps=5)
    assert (wrong < .01).all()
    torch.testing.assert_close(corrected,torch.ones(3))

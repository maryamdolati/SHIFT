import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import pytest
from shift_krno.data import load_darts,load_crypto,load_trajectory,load_irregular
from shift_krno.evaluate import temporal_rollout


def test_darts_time_partition_and_downsample(tmp_path):
    for fn in ('AirPassengers.csv','heart_rate.csv','monthly-sunspots.csv'):
        pd.DataFrame({'date':np.arange(400),'value':np.sin(np.arange(400)/7.)}).drop(columns='date').to_csv(tmp_path/fn,index=False)
    d=load_darts(tmp_path,'AirPassengers',10,5)
    assert d.metadata['train_end']==240 and d.metadata['val_end']==320
    assert all(o+5<=240 for i,o in d.train.origins)
    assert all(o>=240 and o+5<=320 for i,o in d.val.origins)
    pd.DataFrame({'x':np.arange(400,dtype=float)}).to_csv(tmp_path/'monthly-sunspots.csv',index=False)
    assert load_darts(tmp_path,'Sunspots',10,5).metadata['total']==100
    assert load_darts(tmp_path,'HeartRate',10,5).metadata['total']==200


def test_crypto_missing_weights_or_asset_manifest_fail_closed(tmp_path):
    np.save(tmp_path/'train.npy',np.random.default_rng(11).normal(size=(14,240,8)).astype(np.float32))
    np.save(tmp_path/'test.npy',np.random.default_rng(12).normal(size=(14,140,8)).astype(np.float32))
    with pytest.raises(FileNotFoundError):load_crypto(tmp_path)
    np.save(tmp_path/'weights.npy',np.array([4.30406509320417,6.779921907472252,2.3978952727983707,4.406719247264253,3.555348061489413,1.3862943611198906,5.8944028342648505,2.079441541679836,1.0986122886681098,2.3978952727983707,1.0986122886681098,1.6094379124341005,2.079441541679836,1.791759469228055],dtype=np.float32))
    with pytest.raises(FileNotFoundError):load_crypto(tmp_path)
    (tmp_path/'assets.json').write_text(json.dumps({'asset_ids':list(range(14))}))
    b=load_crypto(tmp_path)
    assert b.channels==8
    assert all(b.train[j]['origin'].item()<=180 for j in range(len(b.train)))
    assert all(b.test[j]['origin'].item()>=240+100 for j in range(len(b.test)))


def test_trajectory_all_examples_and_all_five_origins(tmp_path):
    np.save(tmp_path/'train.npy',np.random.randn(3,310,2).astype(np.float32))
    np.save(tmp_path/'test.npy',np.random.randn(7,310,2).astype(np.float32))
    b=load_trajectory(tmp_path)
    assert len(b.series[1])==7*5
    assert {o for i,o in b.series[1]}=={50,100,150,200,250}
    assert b.metadata['test_examples']==7


def test_irregular_requires_provenance(tmp_path):
    (tmp_path/'mujoco').mkdir()
    with pytest.raises(FileNotFoundError):load_irregular(tmp_path,'mujoco')


class UnitMeanForecaster(torch.nn.Module):
    def __init__(self):
        super().__init__();self.dummy=torch.nn.Parameter(torch.tensor(0.))
    def forward(self,x,mask,t,q,K=2,mode='conditional'):
        from types import SimpleNamespace
        m=torch.ones(x.shape[0],q.shape[1],x.shape[2],device=x.device)
        return SimpleNamespace(mean=m,u_value=m*0,u_temporal=m*0)


def test_recursive_rollout_uses_only_its_predictions():
    model=UnitMeanForecaster()
    start=torch.zeros(2,30,2)
    pr,_,_=temporal_rollout(model,start,30,15,30,K=2,origin_time=30)
    assert pr.shape==(2,30,2)
    assert torch.all(pr==1)


def test_60_step_rollout_not_5_step_only():
    model=UnitMeanForecaster()
    x=torch.zeros(1,5,3)
    p,_,_=temporal_rollout(model,x,60,5,5,K=2,origin_time=5)
    assert p.size(1)==60


def test_shallow_full_60_steps_and_all_variables():
    from types import SimpleNamespace
    from shift_krno.evaluate import eval_shallow
    class TrivialSpatial(torch.nn.Module):
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.tensor(0.));self.calls=0
        def forward(self,x,mask,t,q,K=8,mode='conditional'):
            self.calls+=1
            pred=torch.zeros((x.shape[0],q.shape[1],32,32,3),device=x.device)
            return SimpleNamespace(mean=pred,u_temporal=pred,u_value=pred)
    model=TrivialSpatial()
    sample=np.ones((1,65,32,32,3),dtype=np.float32)
    b=SimpleNamespace(series=sample)
    m=eval_shallow(model,b)
    assert m['recursive_steps']==55 and m['model_calls_per_trajectory']==11
    assert model.calls==11
    for k in ('relative_l2_rho','relative_l2_u','relative_l2_v'):
        assert abs(m[k]-1.)<1e-5


def test_m4_no_test_series_silently_omitted(tmp_path):
    from shift_krno.data import load_m4
    (tmp_path/'Train').mkdir();(tmp_path/'Test').mkdir()
    tr=pd.DataFrame([['A']+list(range(40)),['B']+list(range(15))+[np.nan]*25])
    te=pd.DataFrame([['A']+list(range(8)),['B']+list(range(8))])
    tr.to_csv(tmp_path/'Train/Quarterly-train.csv',index=False,header=['V1']+[f'V{i}' for i in range(2,42)])
    te.to_csv(tmp_path/'Test/Quarterly-test.csv',index=False,header=['V1']+[f'V{i}' for i in range(2,10)])
    b=load_m4(tmp_path,'Quarterly',lookback=14,pred=8)
    assert b.metadata['excluded']==0
    from shift_krno.evaluate import eval_m4
    assert eval_m4(UnitMeanForecaster(),b)['n_series']==2
    with pytest.raises(ValueError,match='shorter than'):
        load_m4(tmp_path,'Quarterly',lookback=16,pred=8)

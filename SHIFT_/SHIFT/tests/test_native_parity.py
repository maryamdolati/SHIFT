import ast
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset
from shift_krno.data import load_m4,load_trajectory,load_crypto
from shift_krno.losses import crypto_blocks


@pytest.fixture
def native():
    root=os.environ.get('KRNO_UPSTREAM_ROOT')
    if not root:pytest.skip('Set KRNO_UPSTREAM_ROOT to verify the supplied upstream tree')
    exp=Path(root)/'experiments'
    path=exp/'data_provider/data_loader.py'
    tree=ast.parse(path.read_text());names={'Dataset_M4_KNF','CrytosDataset','TrajDataset'}
    selected=ast.Module(body=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in names],type_ignores=[])
    ns=dict(np=np,torch=torch,Dataset=Dataset,os=os);exec(compile(selected,str(path),'exec'),ns)
    spec=importlib.util.spec_from_file_location('native_metrics',exp/'utils/KNF_eval_metrics.py');metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
    return ns,metric


def compare(ds,reference):
    assert len(ds)==len(reference)
    for j in range(len(ds)):
        x,y,*_=reference[j];r=ds[j]
        np.testing.assert_allclose(r['x'].numpy(),x,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(r['y'].numpy(),y,rtol=1e-6,atol=1e-6)


def test_native_trajectory_loader(native,tmp_path):
    ns,_=native;rng=np.random.default_rng(2)
    np.save(tmp_path/'train.npy',rng.normal(size=(3,300,2)).astype(np.float32));np.save(tmp_path/'test.npy',rng.normal(size=(2,300,2)).astype(np.float32))
    b=load_trajectory(tmp_path)
    for split,d in [('train',b.train),('val',b.val)]:compare(d,ns['TrajDataset'](str(tmp_path),flag=split,size=[30,0,15],jumps=2))
    official=ns['TrajDataset'](str(tmp_path),flag='test',size=[30,0,30],jumps=2)
    for k,(i,o) in enumerate(b.series[1]):
        x,y,*_=official[k];np.testing.assert_allclose(b.series[0][i][o-30:o],x,rtol=1e-6,atol=1e-6);np.testing.assert_allclose(b.series[0][i][o:o+30],y,rtol=1e-6,atol=1e-6)


def test_native_crypto_loader_and_metric(native,tmp_path):
    ns,metric=native;rng=np.random.default_rng(3)
    np.save(tmp_path/'train.npy',rng.normal(size=(14,1000,8)).astype(np.float32));np.save(tmp_path/'test.npy',rng.normal(size=(14,220,8)).astype(np.float32))
    weights=metric.WRMSE.__defaults__[0];np.save(tmp_path/'weights.npy',weights);(tmp_path/'assets.json').write_text(json.dumps({'asset_ids':list(range(14))}))
    b=load_crypto(tmp_path)
    for split,d in [('train',b.train),('val',b.val),('test',b.test)]:compare(d,ns['CrytosDataset'](str(tmp_path),flag=split,size=[80,0,15],jumps=100))
    y=rng.normal(size=(14,3,15,8)).astype(np.float32);p=y+rng.normal(size=y.shape).astype(np.float32)*.01
    exact=metric.WRMSE(p,y);actual=crypto_blocks(torch.tensor(y.transpose(1,0,2,3)),torch.tensor(p.transpose(1,0,2,3)),torch.tensor(weights))
    np.testing.assert_allclose([actual['official_short'],actual['official_medium'],actual['official_long_step'],actual['total']],exact,rtol=1e-6)


def test_native_m4_loader(native,tmp_path):
    ns,_=native;rng=np.random.default_rng(7);train=rng.normal(size=(2,100)).astype(np.float32);test=rng.normal(size=(2,8)).astype(np.float32)
    np.save(tmp_path/'train.npy',{'Quarterly':train});np.save(tmp_path/'test.npy',{'Quarterly':test})
    for kind,values in [('train',train),('test',test)]:
        df=pd.DataFrame(values);df.insert(0,'id',['Q1','Q2']);df.to_csv(tmp_path/f'Quarterly-{kind}.csv',index=False)
    b=load_m4(tmp_path,'Quarterly',14,8)
    for split,d in [('train',b.train),('val',b.val)]:compare(d,ns['Dataset_M4_KNF'](str(tmp_path),flag=split,size=[14,0,8],seasonal_patterns='Quarterly',jumps=2,val_train_ratio=.05))

import numpy as np
import torch
import pytest
from shift_krno.data import official_origins,official_segments,ImplicitWindows,PreprocessedIrregular
from shift_krno.losses import masked_macro_mse,variable_error_sums,macro_from_sums,crypto_blocks
from shift_krno.evaluate import eval_irregular
from types import SimpleNamespace


def test_eligible_window_partition_matches_executable():
    for n,L,P,stride,f in [(100,14,8,2,.05),(1000,80,15,100,.1),(310,30,15,2,.1),(50,30,15,2,.1)]:
        starts=np.arange(0,n-L-P,stride);cut=int((1-f)*len(starts))
        train,val=official_origins(0,n,L,P,stride,f)
        assert [o for _,o in train]==list(starts[:cut]+L)
        assert [o for _,o in val]==list(starts[cut:]+L)
        for a,b in zip(official_segments(0,n,L,P,stride,f),(train,val)):
            ds=ImplicitWindows([np.zeros((n,1),np.float32)],[a] if a else [],L,P)
            assert [ds[j]['origin'].item() for j in range(len(ds))]==[o for _,o in b]


def test_macro_counts_per_variable():
    y=torch.zeros(1,4,2);pred=torch.tensor([[[1.,3.],[1.,0.],[1.,0.],[1.,0.]]]);m=torch.tensor([[[1.,1.],[1.,0.],[1.,0.],[1.,0.]]])
    assert masked_macro_mse(pred,y,m).item()==5.
    se,ae,n=variable_error_sums(pred,y,m)
    assert macro_from_sums(se,n).item()==5. and macro_from_sums(ae,n).item()==2.


def test_irregular_ragged_export_roundtrip(tmp_path):
    p=tmp_path/'data.npz';x=np.arange(10,dtype=np.float32).reshape(5,2);y=np.zeros((3,2),np.float32)
    np.savez(p,x=x,mask=np.ones_like(x),t=np.array([0,.1,0,.2,.3]),q=np.array([.4,.5,.6]),y=y,ymask=np.ones_like(y),x_offsets=[0,2,5],y_offsets=[0,1,3],record_ids=['a','b'])
    ds=PreprocessedIrregular(p,2);assert len(ds)==2
    assert ds[0]['x'].shape==(2,2) and ds[1]['x'].shape==(3,2)
    assert ds[0]['y'].shape==(1,2) and ds[1]['y'].shape==(2,2)


def test_crypto_total_all_horizons_not_last():
    y=torch.zeros(1,14,15,8);p=y.clone();p[:,:,0,-1]=1
    d=crypto_blocks(y,p,torch.ones(14));assert d['total']==pytest.approx(15**-.5);assert d['11-15']==0.


def test_macro_evaluation_across_batches():
    class Zero(torch.nn.Module):
        def __init__(self):super().__init__();self.a=torch.nn.Parameter(torch.tensor(0.))
        def forward(self,x,m,t,q,K,mode):return SimpleNamespace(mean=torch.zeros(x.shape[0],q.shape[1],x.shape[-1]))
    def row(y,m):
        return dict(x=torch.zeros(2,2),mask=torch.ones(2,2),t=torch.tensor([0.,1.]),q=torch.arange(2.,2.+len(y)),y=torch.tensor(y),ymask=torch.tensor(m))
    b=SimpleNamespace(metadata={'dataset':'mimic','variable_length':True},test=[row([[1.,3.]],[[1.,1.]]),row([[1.,0.],[1.,0.]],[[1.,0.],[1.,0.]])])
    q=eval_irregular(Zero(),b);assert q['mse']==5. and q['mae']==2.

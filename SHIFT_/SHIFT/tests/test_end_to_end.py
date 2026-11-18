import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from shift_krno.uncertainty_audit import ProbabilisticAudit


def _fake_data(root):
    p=root/'synthetic';p.mkdir()
    hashes={}
    rng=np.random.default_rng(23)
    for name,n in [('train',8),('val',4),('test',4)]:
        x=rng.normal(size=(n,6,2)).astype('float32')
        t=np.tile(np.array([0,.1,.25,.4,.55,.75],dtype='float32'),(n,1))
        q=np.tile(np.array([.85,.95,1.05],dtype='float32'),(n,1))
        y=x[:,-1:, :]+np.zeros((n,3,2),dtype='float32')
        path=p/f'{name}.npz'
        np.savez_compressed(path,x=x,mask=np.ones_like(x),t=t,q=q,y=y,ymask=np.ones_like(y))
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (p/'manifest.json').write_text(json.dumps(dict(source_repository='synthetic test',
        source_commit='unit-test-only',source_split_hashes=hashes,
        preprocessing_description='generated data with no unseen inputs')))


def test_paired_training_audit_same_data_contract(tmp_path,monkeypatch):

    from test_reviewer_fixes import fake_krno
    from shift_krno.run import main
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from experiments.scripts.shift_review import compare_audits
    (tmp_path/"data").mkdir()
    _fake_data(tmp_path/"data")
    with fake_krno(monkeypatch):
        for arch in ('krno_reference','krno_warp'):
            main(['--dataset','synthetic','--data-root',str(tmp_path/'data'),'--out',str(tmp_path/'runs'/arch),
                  '--architecture',arch,'--epochs','2','--batch-size','4',
                  '--train-samples','2','--eval-samples','2','--test-forbidden'])
    a=json.loads((tmp_path/'runs/krno_warp/audit.json').read_text())
    b=json.loads((tmp_path/'runs/krno_reference/audit.json').read_text())
    r=compare_audits.check(a,b,comparison='operator_matched')
    assert r['passed'],r
    assert a['data_contract']==b['data_contract']
    assert a['data_inventory']==b['data_inventory']


def test_mixture_calibration_mask_and_time_uncertainty():
    from types import SimpleNamespace
    K,B,H,C=3,2,4,2
    pred=torch.zeros(K,B,H,C)
    pred[0]+=0.3;pred[2]-=0.2
    vari=torch.ones_like(pred)*0.5
    out=SimpleNamespace(component_mean=pred,component_variance=vari)
    target=torch.zeros(B,H,C)
    mask=torch.ones_like(target)
    mask[:,0]=0
    acc=ProbabilisticAudit();acc.update(out,target,mask)
    res=acc.result()
    assert res['scalar_observations']==B*(H-1)*C
    assert 0<=res['coverage95']<=1
    assert res['u_temporal']>0
    assert 'calibration_error' in res
    assert 'interval_score95' in res

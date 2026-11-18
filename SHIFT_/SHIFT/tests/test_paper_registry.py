import importlib.util
import json
from pathlib import Path
import pytest


def launcher():
    spec=importlib.util.spec_from_file_location('paper_runner',Path(__file__).parents[1]/'run_paper_protocols.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('dataset,seeds',[('mimic',[1,2,3,4,5]),('ushcn',[1,2,3,4,5]),('activity',[1,2,3,4,5]),('crypto',[2021,2024,2025,2027]),('trajectory',[2021,2024,2025,2027]),('m4/Weekly',[2021,2023,2025,2030]),('m4/Daily',[2021,2023,2025,2030]),('mujoco',[0,1,2,3,4]),('shallow',[42])])
def test_seed_provenance(dataset,seeds):
    assert launcher().REGISTRY[dataset]['seeds']==seeds


def test_mimic_plan_separates_native_and_full_shift(tmp_path):
    r=launcher();r.main(['--dataset','mimic','--data-root',str(tmp_path/'data'),'--upstream-root',str(tmp_path/'up'),'--out',str(tmp_path),'--plan-only'])
    q=json.loads((tmp_path/'locked_paper_mimic.json').read_text());assert len(q['commands'])==10
    assert {x['architecture'] for x in q['commands']}=={'krno_official','shift'}
    shift=next(x['command'] for x in q['commands'] if x['architecture']=='shift')
    assert shift[shift.index('--lr')+1]=='0.0003'
    assert shift[shift.index('--grad-accum')+1]=='16'
    assert '--shape-penalty' in shift and '--allow-reconstructed-objective' in shift
    assert q['blockers'] and not q['reproduction_verified']


def test_locked_seed_and_reconstruction_gate(tmp_path):
    r=launcher();base=['--dataset','mimic','--data-root',str(tmp_path/'data'),'--upstream-root',str(tmp_path/'up'),'--out',str(tmp_path)]
    with pytest.raises(ValueError,match='seed list'):r.main(base+['--seeds','42'])
    with pytest.raises(ValueError,match='Reconstruction not executable'):r.main(base+['--execute'])


def test_missing_rates_separate_paths(tmp_path):
    r=launcher();base=['--dataset','mujoco','--data-root',str(tmp_path/'data'),'--upstream-root',str(tmp_path/'up'),'--out',str(tmp_path),'--comparison','warp-only']
    for rate in ('.3','.7'):r.main(base+['--missing-rate',rate])
    a=json.loads((tmp_path/'locked_warp-only_mujoco_miss_0.3.json').read_text());b=json.loads((tmp_path/'locked_warp-only_mujoco_miss_0.7.json').read_text())
    assert a['commands'][0]['output']!=b['commands'][0]['output']
    cmd=a['commands'][0]['command'];assert 'seed_0' in cmd[cmd.index('--data-root')+1]

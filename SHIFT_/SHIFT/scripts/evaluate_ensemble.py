import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'krno'))
from shift_krno.ensemble import FixedCoordinateEnsemble
from shift_krno.run import create_model, evaluate, file_inventory, load_dataset, profile, upstream_identity


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--upstream-root',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if len(a.checkpoints)<2:raise ValueError('Supply at least two independently trained fixed-coordinate checkpoints')
    if len({c.resolve() for c in a.checkpoints})!=len(a.checkpoints):raise ValueError('Duplicate checkpoints')
    if a.out.exists():raise FileExistsError(a.out)
    identity=upstream_identity(a.upstream_root)
    cps=[torch.load(c,map_location='cpu',weights_only=False) for c in a.checkpoints]
    records=[c['metadata'] for c in cps]
    arguments=[r['arguments'] for r in records]
    keys=('dataset','architecture','backbone','mode','probabilistic','objective','lookback','block','width','kernel_hidden','hidden','integral_layers','revin','val_fraction','train_stride','test_stride','missing_rate','epochs','batch_size','grad_accum','lr','weight_decay','optimizer','scheduler','scheduler_factor','scheduler_patience','patience','clip','nll_weight','reg_strength','input_noise','dropout','train_samples','eval_samples','strength')
    if any(any(v.get(k)!=arguments[0].get(k) for k in keys) for v in arguments[1:]):raise ValueError('Ensemble members must use the same locked training and data protocol')
    seeds=[v['seed'] for v in arguments]
    if len(set(seeds))!=len(seeds):raise ValueError('Ensemble members require distinct training seeds')
    if any(v.get('mode')!='fixed' or v.get('backbone')!='original' or v.get('objective') not in ('mix_nll','component_nll','mse_plus_nll') or (v.get('objective')=='mse_plus_nll' and v.get('nll_weight',0)<=0) or v.get('test') for v in arguments):raise ValueError('Use fixed-coordinate likelihood-trained checkpoints selected without test evaluation')
    inventory=file_inventory(a.data_root)
    if any(r.get('data_inventory')!=inventory for r in records):raise ValueError('Ensemble data differ from the frozen member data')
    if any(r.get('upstream_KRNO',{}).get('commit')!=identity.get('commit') for r in records):raise ValueError('Original operator source identity differs')
    cfg=SimpleNamespace(**arguments[0]);cfg.data_root=a.data_root;cfg.upstream_root=a.upstream_root;cfg.test_forbidden=False;cfg.test=True;cfg.probabilistic_audit=True;cfg.eval_samples=len(cps)
    if cfg.dataset=='mujoco':raise ValueError('For MuJoCo, use independent training seeds on one frozen preprocessing seed/rate; the current native registry binds these seeds together')
    bundle=load_dataset(cfg,refit=cfg.refit)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    members=[]
    for checkpoint in cps:
        model=create_model(cfg,bundle.channels).to(device)
        model.load_state_dict(checkpoint['weights']);model.has_predictive_variance=True;model.inference_strength=cfg.strength;model.eval();members.append(model)
    ensemble=FixedCoordinateEnsemble(members).to(device)
    result=evaluate(ensemble,bundle,cfg)
    record=dict(dataset=cfg.dataset,seeds=seeds,member_parameter_counts=[profile(m) for m in members],members=[str(c) for c in a.checkpoints],variance_semantics='Between-member uncertainty, not between-warp geometry; evaluator temporal-variance keys refer to member disagreement only',result=result,source=identity)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()

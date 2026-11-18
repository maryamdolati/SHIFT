import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REG=json.loads((ROOT/'protocol_registry.json').read_text())
ARMS={
    'shift':('shift','conditional',{}),
    'fixed':('shift','fixed',{}),
    'deterministic':('shift','deterministic',{}),
    'random':('shift','random',{}),
    'unconditional':('shift','unconditional',{}),
    'input_noise':('shift','fixed',{'input_noise':None}),
    'dropout':('shift','fixed',{'dropout':None}),
    'krno_control':('krno_reference','fixed',{'probabilistic':True}),
    'krno_warp_control':('krno_warp','conditional',{'probabilistic':True}),
}


def flags(cfg):
    items=[]
    for k,v in cfg.items():
        if v is None:continue
        f='--'+k.replace('_','-')
        if isinstance(v,bool):
            if v:items.append(f)
        else:items.extend([f,str(v)])
    return items


def create_plan(dataset,data_root,upstream_root,out,config,missing=None,test=False):
    if dataset not in REG:raise ValueError('Unknown benchmark')
    if dataset=='mujoco' and missing not in (0.,.3,.5,.7):raise ValueError('Provide missing rate')
    required={'epochs','patience','batch_size','grad_accum','lr','weight_decay','width','kernel_hidden','train_samples','eval_samples','input_noise','dropout','objective','capacity_widths'}
    if required-set(config):raise ValueError(f'Missing preregistered settings: {sorted(required-set(config))}')
    if config['objective'] not in ('mix_nll','mse_plus_nll'):raise ValueError('All arms must train a variance head using a shared likelihood objective')
    if min(config['train_samples'],config['eval_samples'])<2:raise ValueError('Use K>=2')
    if config['objective']=='mse_plus_nll' and config.get('nll_weight',0)<=0:raise ValueError('Positive shared NLL weight required')
    if not config['input_noise']>0 or not 0<config['dropout']<1:raise ValueError('Predeclare nonzero input noise and dropout')
    if 'seeds' in config or 'mode' in config or 'architecture' in config or 'reg_strength' in config:raise ValueError('These controls are locked')
    if dataset in ('mimic','ushcn','activity','mujoco') and config['batch_size']!=1:raise ValueError('Variable-length examples require batch size one')
    if dataset=='shallow' and config.get('block',5)!=5:raise ValueError('Five-step blocks required')
    common={k:v for k,v in config.items() if k not in ('input_noise','dropout','capacity_widths')}
    if common.get('revin'):raise ValueError('Use a common raw-operator configuration without RevIN, or implement it identically for all arms')
    cfg=REG[dataset];seeds=cfg['seeds'];tag=dataset.replace('/','_')+(f'_miss_{missing:g}' if dataset=='mujoco' else '')
    items=[]
    for seed,(arm,(arch,mode,additional)) in itertools.product(seeds,ARMS.items()):
        data=Path(data_root).resolve()/cfg['data_dir']
        if dataset=='mujoco':data=Path(data_root).resolve()/f'irregular_missing_{missing:g}'/f'seed_{seed}'
        target=Path(out).resolve()/tag/f'seed_{seed}'/arm
        settings=dict(common,architecture=arch,mode=mode,reg_strength=0,**{k:config[k] if v is None else v for k,v in additional.items()})
        cmd=[sys.executable,'-m','shift_krno.run','--dataset',dataset,'--data-root',str(data),'--upstream-root',str(Path(upstream_root).resolve()),'--seed',str(seed),'--out',str(target),*flags(settings)]
        if dataset=='mujoco':cmd+=['--missing-rate',str(missing)]
        if test:
            cp=target/'best.pt'
            if not cp.is_file():raise FileNotFoundError(f'Missing validation-selected checkpoint: {cp}')
            cmd+=['--test','--probabilistic-audit','--checkpoint',str(cp)]
            cmd[cmd.index('--out')+1]=str(target/'final_test')
        else:cmd+=['--test-forbidden']
        items.append(dict(seed=seed,arm=arm,command=cmd,output=cmd[cmd.index('--out')+1]))
        if arm=='shift' and test:
            audit_cmd=cmd.copy();audit_cmd[audit_cmd.index('--out')+1]=str(target.parent/'donor');audit_cmd+=['--donor-audit']
            items.append(dict(seed=seed,arm='independent_donor',command=audit_cmd,output=str(target.parent/'donor')))
    return dict(dataset=dataset,protocol='controlled reconstruction; not historical paper rerun',shared_objective=config['objective'],seeds=seeds,capacity_grid=config['capacity_widths'],capacity_matching_verified=False,commands=items)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=sorted(REG));p.add_argument('--data-root',required=True);p.add_argument('--upstream-root',required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',default='runs/reviewer');p.add_argument('--missing-rate',type=float);p.add_argument('--test',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    plan=create_plan(a.dataset,a.data_root,a.upstream_root,a.out,json.loads(a.config.read_text()),a.missing_rate,a.test)
    output=Path(a.out);output.mkdir(parents=True,exist_ok=True);tag=a.dataset.replace('/','_')+(f'_miss_{a.missing_rate:g}' if a.dataset=='mujoco' else '')
    path=output/f'{tag}_reviewer_plan.json';path.write_text(json.dumps(plan,indent=2)+'\n');print(path)
    if a.execute:
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'krno')+os.pathsep+env.get('PYTHONPATH','')
        for item in plan['commands']:subprocess.run(item['command'],check=True,cwd=ROOT,env=env)


if __name__=='__main__':main()

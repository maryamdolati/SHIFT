import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REGISTRY=json.loads((ROOT/'protocol_registry.json').read_text())
IRREGULAR={'mimic','ushcn','activity'}


def native_command(root,name,seed,missing,gpu):
    exp=root/'experiments'
    if name in IRREGULAR:
        cwd=exp/'scripts/irregular_time_series/krno'
        return cwd,[sys.executable,'run_models_new.py','--dataset',name,'--state','def','--history',str(3000 if name=='activity' else 24),'--epoch','100','--patience','10','--batch_size','1','--lr','0.001','--seed',str(seed),'--gpu',str(gpu),'--int_channels','20','--accumulation_steps','16']
    if name=='mujoco':
        return exp/'scripts/mujoco',[sys.executable,'run_mujoco.py','--seed',str(seed),'--lr','.001','--missing_rate',str(missing),'--time_seq','50','--y_seq','10','--epoch','500','--step_mode','valloss','--model','krno']
    if name=='shallow':
        return exp/'scripts/shallow_water_krno',[sys.executable,'train_script.py','--lag','5']
    if name=='m4/Quarterly':
        return exp,None
    script={'crypto':'run_crypto_KRNO.sh','trajectory':'run_traj_KRNO.sh','m4/Weekly':'run_m4_KRNO.sh','m4/Daily':'run_m4_KRNO.sh'}[name]
    lines=(exp/'scripts/m4_crypto_traj'/script).read_text().splitlines()
    for line in lines:
        if not line.strip().startswith('python'):continue
        cmd=shlex.split(line)
        if name.startswith('m4/') and cmd[cmd.index('--seasonal_patterns')+1]!=name.split('/')[1]:continue
        if not name.startswith('m4/') and cmd[cmd.index('--seed')+1]!=str(seed):continue
        cmd[0]=sys.executable;cmd[cmd.index('--seed')+1]=str(seed)
        if '--gpu' not in cmd:cmd+=['--gpu',str(gpu)]
        return exp,cmd
    raise ValueError('Native selected command not found')


def flags(config):
    result=[]
    for k,v in config.items():
        flag='--'+k.replace('_','-')
        if isinstance(v,bool):
            if v:result.append(flag)
        elif v is not None:result.extend([flag,str(v)])
    return result


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=sorted(REGISTRY));p.add_argument('--comparison',choices=['paper','official-krno','shift','warp-only'],default='paper');p.add_argument('--data-root');p.add_argument('--upstream-root');p.add_argument('--out',default='runs/paper_protocol');p.add_argument('--missing-rate',type=float);p.add_argument('--seeds',type=int,nargs='+');p.add_argument('--gpu',default='0');p.add_argument('--shift-config',type=Path);p.add_argument('--test',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--plan-only',action='store_true');p.add_argument('--allow-reconstruction',action='store_true');a=p.parse_args(argv)
    if a.execute and a.plan_only:raise ValueError('Choose execution or plan only')
    if a.dataset=='mujoco' and a.missing_rate not in (0.,.3,.5,.7):raise ValueError('MuJoCo needs the official missing rate')
    if a.dataset!='mujoco' and a.missing_rate is not None:raise ValueError('Missing rate is MuJoCo-only')
    c=REGISTRY[a.dataset];seeds=c['seeds']
    if a.seeds is not None and a.seeds!=seeds:raise ValueError('Official seed list locked')
    if not a.upstream_root:raise ValueError('Provide --upstream-root')
    up=Path(a.upstream_root).expanduser().resolve();out=Path(a.out).resolve();tag=a.dataset.replace('/','_')+(f'_miss_{a.missing_rate:g}' if a.dataset=='mujoco' else '')
    extra=json.loads(a.shift_config.read_text()) if a.shift_config else {}
    cfg=c['shift_profile'].copy()
    forbidden=set(extra)&set(cfg)
    if forbidden and any(extra[k]!=cfg[k] for k in forbidden):raise ValueError(f'Cannot silently override documented manuscript settings: {sorted(forbidden)}')
    cfg.update(extra)
    missing=[k for k in c['required_original_settings'] if k not in cfg]
    items=[];blockers=[]
    if a.comparison in ('paper','shift'):
        if missing:blockers.append('Missing original SHIFT settings: '+', '.join(missing))
        if a.dataset=='activity':blockers.append('Original Activity pretraining/fine-tuning checkpoint procedure not supplied')
        if a.dataset=='mujoco':blockers.append('Original per-seed MuJoCo get_data_krno exports must be verified; external dependency absent from uploaded source')
        if a.dataset=='trajectory':blockers.append('Manuscript states 15-step test, native test is 30; confirm historical runs and correct manuscript')
        if a.dataset=='ushcn':blockers.append('Manuscript states 24-month future, native code uses 1 month; confirm historical runs')
        if a.dataset=='m4/Quarterly':blockers.append('Native final selected Quarterly configuration not published; choose validation-selected native configuration independently')
    if a.comparison!='official-krno' and not a.data_root:raise ValueError('Provide --data-root')
    for seed in seeds:
        dest=out/tag/f'seed_{seed}'
        if a.comparison in ('official-krno','paper'):
            cwd,cmd=native_command(up,a.dataset,seed,a.missing_rate,a.gpu)
            if cmd is None:blockers.append('Native Quarterly selected command unavailable; run original validation tuning');
            else:
                items.append(dict(seed=seed,architecture='krno_official',cwd=str(cwd),command=cmd,output='Native upstream output directories'))
                if a.dataset=='shallow' and a.test:
                    items.append(dict(seed=seed,architecture='krno_official_rollout',cwd=str(cwd),command=[sys.executable,'post_process.py','--file','ckpts/krno_it115000_42.pt'],output='Native upstream output directories'))
        if a.comparison in ('paper','shift','warp-only'):
            data=Path(a.data_root).resolve()/c['data_dir']
            if a.dataset=='mujoco':data=Path(a.data_root).resolve()/f'irregular_missing_{a.missing_rate:g}'/f'seed_{seed}'
            arms=[('shift','conditional')] if a.comparison!='warp-only' else [('krno_reference','fixed'),('krno_warp','conditional')]
            for arch,mode in arms:
                settings=cfg.copy() if arch=='shift' else {k:c[k] for k in ('epochs','batch_size','lr','width','kernel_hidden','lookback','block','train_stride','test_stride') if k in c}
                if arch!='shift' and a.dataset in IRREGULAR:settings.update(batch_size=1,grad_accum=16,patience=10,scheduler_factor=.8,scheduler_patience=3)
                if arch!='shift' and a.dataset=='mujoco':settings['batch_size']=1
                where=dest/arch;cmd=[sys.executable,'-m','shift_krno.run','--dataset',a.dataset,'--data-root',str(data),'--seed',str(seed),'--architecture',arch,'--mode',mode,'--out',str(where),'--upstream-root',str(up)]+flags(settings)
                if a.dataset=='mujoco':cmd+=['--missing-rate',str(a.missing_rate)]
                cmd+=['--test' if a.test else '--test-forbidden']
                if arch=='shift':cmd.append('--allow-reconstructed-objective')
                items.append(dict(seed=seed,architecture=arch,cwd=str(ROOT),command=cmd,output=str(where)))
    plan=dict(dataset=a.dataset,seeds=seeds,seed_provenance=c['seed_source'],comparison=a.comparison,missing_rate=a.missing_rate,commands=items,blockers=list(dict.fromkeys(blockers)),reproduction_verified=False)
    out.mkdir(parents=True,exist_ok=True);path=out/f'locked_{a.comparison}_{tag}.json';path.write_text(json.dumps(plan,indent=2)+'\n');print(path)
    if a.execute:
        if a.comparison in ('paper','shift') and (missing or not a.allow_reconstruction):raise ValueError('Reconstruction not executable without original values and --allow-reconstruction; inspect plan')
        if a.comparison=='paper' and blockers:raise ValueError('Unresolved paper comparison blockers; do not run as matched reproduction')
        if a.comparison=='official-krno' and any('Quarterly' in x for x in blockers):raise ValueError('Native Quarterly selection is undocumented')
        for item in items:
            env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'krno')+os.pathsep+env.get('PYTHONPATH','');env['CUDA_VISIBLE_DEVICES']=str(a.gpu) if item['architecture']!='krno_official' else env.get('CUDA_VISIBLE_DEVICES',str(a.gpu))
            subprocess.run(item['command'],cwd=item['cwd'],env=env,check=True)


if __name__=='__main__':main()

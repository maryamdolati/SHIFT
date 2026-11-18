import argparse
import json
from pathlib import Path
import subprocess
import sys


def plans(a):
    common=[sys.executable,'-m','shift_krno.run','--dataset',a.dataset,
            '--data-root',str(a.data_root),'--epochs',str(a.epochs),
            '--batch-size',str(a.batch_size),'--width',str(a.width),
            '--kernel-hidden',str(a.kernel_hidden),
            '--integral-layers',str(a.integral_layers),
            '--objective',a.objective,'--reg-strength','0',
            '--train-samples',str(a.train_samples),'--eval-samples',str(a.eval_samples)]
    for n,v in (('lookback',a.lookback),('block',a.block)):
        if v is not None:common.extend(('--'+n,str(v)))
    if a.probabilistic:
        common.append('--probabilistic')
    if a.probabilistic_audit:
        common.append('--probabilistic-audit')
    if a.test:
        if not a.upstream_root:
            raise ValueError('--test requires a clean pinned --upstream-root')
        common.extend(['--test','--upstream-root',str(a.upstream_root)])
    for seed in a.seeds:
        for arch,mode in (('krno_reference','fixed'),('krno_warp','conditional')):
            dest=Path(a.out)/f'seed_{seed}'/arch
            yield seed,arch,dest,common+['--seed',str(seed),
                 '--architecture',arch,'--mode',mode,'--out',str(dest)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--upstream-root',help='clean pinned ShiftingTime checkout required for --test')
    p.add_argument('--out',required=True)
    p.add_argument('--seeds',type=int,nargs='+',required=True,help='Must be passed from protocol_registry.json, never an arbitrary default')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lookback',type=int)
    p.add_argument('--block',type=int)
    p.add_argument('--width',type=int,default=20)
    p.add_argument('--kernel-hidden',type=int,default=32)
    p.add_argument('--integral-layers',type=int,default=3)
    p.add_argument('--objective',choices=('mse','mix_nll','component_nll','mse_plus_nll'),default='mse')
    p.add_argument('--train-samples',type=int,default=4)
    p.add_argument('--eval-samples',type=int,default=16)
    p.add_argument('--probabilistic',action='store_true')
    p.add_argument('--probabilistic-audit',action='store_true')
    p.add_argument('--test',action='store_true',help='Run test ONLY on already fixed protocol, never tune on it')
    p.add_argument('--plan-only',action='store_true',help='Write plan JSON without running or accessing test')
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds):
        raise ValueError('Seed list contains duplicates')
    if a.probabilistic_audit and not a.probabilistic:
        raise ValueError('--probabilistic-audit requires --probabilistic')
    plan=list(plans(a))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'locked_plan.json').write_text(json.dumps(dict(
        dataset=a.dataset,source_data=str(Path(a.data_root).resolve()),
        seeds=a.seeds,comparison='same operator configuration, warp intervention only',
        method='predeclared paired experimental protocol, identical training budget',
        test_requested=a.test,
        commands=[dict(seed=seed,arch=arch,cmd=cmd) for seed,arch,_,cmd in plan]
    ),indent=2))
    if a.plan_only:
        print('Wrote plan to',out/'locked_plan.json');return
    for seed,arch,dest,cmd in plan:
        subprocess.run(cmd,check=True)
    for seed in a.seeds:
        base=out/f'seed_{seed}'
        cmd=[sys.executable,str(Path(__file__).with_name('compare_audits.py')),
             '--shift',str(base/'krno_warp/audit.json'),
             '--krno-control',str(base/'krno_reference/audit.json'),
             '--comparison','operator_matched']
        subprocess.run(cmd,check=True)
    print('Completed paired code runs. Scientific claims still require official data and external KRNO replication.')


if __name__=='__main__':main()

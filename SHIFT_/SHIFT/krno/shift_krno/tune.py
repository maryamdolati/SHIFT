import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

M4_LENGTHS={
    'Yearly':([3,6],[1,2]),
    'Quarterly':([8,12,14],[4,8]),
    'Monthly':([18,40],[6,12,18]),
    'Weekly':([30,45,60],[6,10,13]),
    'Daily':([18,30,45],[6,10,14]),
    'Hourly':([48,96,192],[5,10]),
}


def candidate_grid(dataset):
    common=dict(lr=[.001,.005],width=[16,32],kernel_hidden=[32,64],revin=[False,True])
    if dataset.startswith('darts/'):
        common['width']=[5,10,32]
        combos=[]
        for lr,w,h,l,p,r in itertools.product(common['lr'],common['width'],
                  common['kernel_hidden'],[10,24,48,96],['5','half','full'],common['revin']):
            pp=5 if p=='5' else l//2 if p=='half' else l
            combos.append(dict(lr=lr,width=w,kernel_hidden=h,lookback=l,block=pp,revin=r))
        return combos
    if dataset.startswith('m4/'):
        name=dataset.split('/',1)[1]
        Ls,Ps=M4_LENGTHS[name]
    elif dataset=='crypto':Ls,Ps=[63,80],[5,10,15]
    elif dataset=='trajectory':Ls,Ps=[10,21,30],[15,30]
    else:raise ValueError('Grid search specified only for KRNO Darts/M4/Crypto/Trajectory')
    common['lr']=[.001]
    return [dict(lr=lr,width=w,kernel_hidden=h,lookback=l,block=p,revin=r)
        for lr,w,h,l,p,r in itertools.product(common['lr'],common['width'],
        common['kernel_hidden'],Ls,Ps,common['revin'])]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True)
    ap.add_argument('--data-root',required=True)
    ap.add_argument('--out',default='runs/grid')
    ap.add_argument('--architecture',choices=['krno_reference','krno_warp','shift','krno_control'],required=True,help='Explicit model choice: this grid originated from KRNO tuning conventions')
    ap.add_argument('--max-trials',type=int,default=0,
                    help='0=full grid; >0=deterministic prefix (state reduction in reports)')
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--seed',type=int,default=2021)
    ap.add_argument('--train-samples',type=int,default=4)
    ap.add_argument('--probabilistic',action='store_true')
    ap.add_argument('--eval-samples',type=int,default=16)
    ap.add_argument('--tune-weight-decay',action='store_true',
                    help='After selecting Darts architecture, independently tune original KRNO WD grid')
    ap.add_argument('--batch-size',type=int,default=512)
    ap.add_argument('--grad-accum',type=int,default=1)
    a=ap.parse_args()
    if a.dataset.startswith('darts/') or a.dataset not in ('crypto','trajectory','m4/Quarterly','m4/Weekly','m4/Daily'):raise ValueError('Only manuscript benchmarks with official grid definitions are supported')
    grid=candidate_grid(a.dataset)
    if a.architecture in ('krno_reference','krno_warp'):
        grid=[dict(c,revin=False) for c in grid if not c['revin']]
    if a.max_trials>0:grid=grid[:a.max_trials]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    scored=[]
    for i,c in enumerate(grid):
        dest=out/f'trial_{i:04d}'
        cmd=[sys.executable,'-m','shift_krno.run','--dataset',a.dataset,
             '--data-root',a.data_root,'--out',str(dest),
             '--architecture',a.architecture,'--epochs',str(a.epochs),
             '--seed',str(a.seed),'--test-forbidden','--batch-size',str(a.batch_size),'--grad-accum',str(a.grad_accum),
             '--train-samples',str(a.train_samples),'--eval-samples',str(a.eval_samples)]
        if a.probabilistic:cmd.append('--probabilistic')
        for key in ('lr','width','kernel_hidden','lookback','block'):
            cmd+=['--'+key.replace('_','-'),str(c[key])]
        if c['revin'] and a.architecture not in ('krno_reference','krno_warp'):cmd+=['--revin']
        subprocess.run(cmd,check=True)
        record=json.loads((dest/'audit.json').read_text())
        scored.append(dict(params=c,validation=record['training']['best_validation_metric'],
                           path=str(dest)))
        (out/'validation_trials.json').write_text(json.dumps(scored,indent=2))
    scored.sort(key=lambda s:s['validation'])
    (out/'selected_by_validation.json').write_text(json.dumps(dict(
        dataset=a.dataset,architecture=a.architecture,budget=len(grid),
        full_KRNO_grid_size=len(candidate_grid(a.dataset)), search_type='local architecture grid; native KRNO search must use upstream script',
        winners=scored[:3],test_used_in_selection=False),indent=2))
    print('Selected purely on validation:',json.dumps(scored[0],indent=2))
    if a.tune_weight_decay:
        if not a.dataset.startswith('darts/'):
            raise ValueError('Separate weight-decay stage is KRNO Darts-specific')
        wd_candidates=[2.0,1.0,0.5,0.1,0.075,0.05,0.025,0.01,0.005]
        winner=scored[0]['params']
        wd_trials=[]
        for i,wd in enumerate(wd_candidates):
            dest=out/f'wd_trial_{i:02d}'
            cmd=[sys.executable,'-m','shift_krno.run',
                 '--dataset',a.dataset,'--data-root',a.data_root,
                 '--out',str(dest),'--architecture',a.architecture,
                 '--epochs',str(a.epochs),'--seed',str(a.seed),
                 '--test-forbidden','--weight-decay',str(wd),
                 '--train-samples',str(a.train_samples),
                 '--eval-samples',str(a.eval_samples)]
            for key in ('lr','width','kernel_hidden','lookback','block'):
                cmd+=['--'+key.replace('_','-'),str(winner[key])]
            if a.probabilistic:cmd.append('--probabilistic')
            if winner['revin'] and a.architecture not in ('krno_reference','krno_warp'):cmd.append('--revin')
            subprocess.run(cmd,check=True)
            log=json.loads((dest/'audit.json').read_text())
            wd_trials.append(dict(wd=wd,validation=log['training']['best_validation_metric']))
        wd_trials.sort(key=lambda r:r['validation'])
        (out/'selected_weight_decay.json').write_text(json.dumps(dict(
            trial_count=len(wd_trials),trials=wd_trials,
            best_weight_decay=wd_trials[0]['wd'],
            method='independent retraining on same train/validation split',
            no_test_used=True),indent=2))
        print('Best WD by validation:',wd_trials[0])

if __name__=='__main__':main()

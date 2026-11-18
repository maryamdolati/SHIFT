import argparse
import json
from pathlib import Path
import subprocess
import sys

CONTROLS=(
 ('reference','krno_reference','fixed',[]),
 ('conditioned','krno_warp','conditional',[]),
 ('deterministic','krno_warp','deterministic',[]),
 ('random_warp','krno_warp','random',[]),
 ('unconditional_warp','krno_warp','unconditional',[]),
 ('shuffled_realized_warp','krno_warp','shuffled',[]),
 ('fixed_warp','krno_warp','fixed',[]),
 ('input_noise','krno_reference','fixed',['--input-noise','0.05']),
 ('input_dropout','krno_reference','fixed',['--dropout','0.05']),
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--width',type=int,default=20)
    p.add_argument('--kernel-hidden',type=int,default=32)
    p.add_argument('--objective',default='mse')
    p.add_argument('--probabilistic',action='store_true')
    p.add_argument('--train-samples',type=int,default=4)
    p.add_argument('--eval-samples',type=int,default=16)
    p.add_argument('--lookback',type=int)
    p.add_argument('--block',type=int)
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    plans=[]
    for seed in a.seeds:
        for label,arch,mode,extra in CONTROLS:
            dst=out/f'seed_{seed}'/label
            cmd=[sys.executable,'-m','shift_krno.run',
                '--dataset',a.dataset,'--data-root',a.data_root,
                '--out',str(dst),'--seed',str(seed),
                '--epochs',str(a.epochs),'--batch-size',str(a.batch_size),
                '--width',str(a.width),'--kernel-hidden',str(a.kernel_hidden),
                '--objective',a.objective,'--train-samples',str(a.train_samples),
                '--eval-samples',str(a.eval_samples),
                '--architecture',arch,'--mode',mode,'--reg-strength','0',
                '--test-forbidden']+extra
            if a.probabilistic:cmd+=['--probabilistic']
            if a.lookback:cmd+=['--lookback',str(a.lookback)]
            if a.block:cmd+=['--block',str(a.block)]
            plans.append(dict(seed=seed,control=label,cmd=cmd))
    (out/'locked_ablation_plan.json').write_text(json.dumps(dict(
        dataset=a.dataset,seeds=a.seeds,trained_separately=True,
        fixed_hyperparameters=True,held_out_test_used=False,
        warning='Shuffled mode needs every validation/training mini-batch to have >=2 samples',
        plans=plans),indent=2))
    if not a.plan_only:
        for r in plans:subprocess.run(r['cmd'],check=True)
    print('Ablation plan:',out/'locked_ablation_plan.json')

if __name__=='__main__':main()

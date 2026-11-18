import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'krno'))
from shift_krno.run import create_model,parser,profile


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--channels',type=int,required=True);p.add_argument('--shift-width',type=int,required=True);p.add_argument('--shift-kernel-hidden',type=int,required=True);p.add_argument('--widths',type=int,nargs='+',required=True);p.add_argument('--kernel-widths',type=int,nargs='+',required=True);p.add_argument('--tolerance',type=float,default=.05);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    args=parser().parse_args(['--dataset',a.dataset,'--data-root','.','--architecture','shift','--width',str(a.shift_width),'--kernel-hidden',str(a.shift_kernel_hidden)])
    target=profile(create_model(args,a.channels))['trainable_params'];candidates=[]
    args.architecture='krno_reference';args.probabilistic=True
    for w in a.widths:
        for h in a.kernel_widths:
            args.width=w;args.kernel_hidden=h
            n=profile(create_model(args,a.channels))['trainable_params']
            candidates.append(dict(width=w,kernel_hidden=h,parameters=n,relative_gap=abs(n-target)/target))
    candidates.sort(key=lambda r:r['relative_gap']);best=candidates[0]
    record=dict(shift_parameters=target,tolerance=a.tolerance,candidates=candidates,selected=best,matched=best['relative_gap']<=a.tolerance,selection='parameter counts only; no labels or scores',baseline='local KRNO control, not native official KRNO')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    if not record['matched']:raise ValueError('No candidate meets preregistered capacity tolerance')


if __name__=='__main__':main()

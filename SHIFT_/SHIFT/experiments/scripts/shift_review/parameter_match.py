import argparse
import json
from shift_krno.model import KRNOReference,KRNOWarpOnly,SpatialKRNOReference,SpatialKRNOWarpOnly


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--channels',type=int,default=8)
    p.add_argument('--spatial',action='store_true')
    p.add_argument('--shift-width',type=int,default=20)
    p.add_argument('--krno-width',type=int,default=20)
    p.add_argument('--tolerance',type=float,default=.10)
    p.add_argument('--out',default='parameter_match.json')
    a=p.parse_args()
    shift=(SpatialKRNOWarpOnly(width=a.shift_width) if a.spatial else
           KRNOWarpOnly(a.channels,width=a.shift_width))
    base=(SpatialKRNOReference(width=a.krno_width) if a.spatial else
          KRNOReference(a.channels,width=a.krno_width))
    n=lambda model:sum(p.numel() for p in model.parameters() if p.requires_grad)
    count_s=n(shift);count_b=n(base)
    ratio=count_s/count_b
    report=dict(shift_params=count_s,krno_control_params=count_b,ratio=ratio,
                matched=(1/(1+a.tolerance)<=ratio<=(1+a.tolerance)),
                same_upstream_operator_package=True)
    with open(a.out,'w') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    if not report['matched']:
        raise SystemExit('Parameter budgets unmatched: adjust widths or declare size-confounded comparison')

if __name__=='__main__':main()

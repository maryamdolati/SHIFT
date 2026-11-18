import argparse
import json
from pathlib import Path
import numpy as np
import torch
from shift_krno.data import load_irregular
from shift_krno.model import ConditionalWarp


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=16)
    a=p.parse_args()
    bundle=load_irregular(a.data_root,'synthetic')
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    args=cp['metadata']['arguments']
    if args['architecture']!='krno_warp':
        raise ValueError('A learned warp-only checkpoint is required')
    model=ConditionalWarp(bundle.channels,hidden=32)
    model.load_state_dict({k[len('warp.'):]:v for k,v in cp['weights'].items()
                           if k.startswith('warp.')})
    model.eval()
    ds=bundle.test
    manifest=bundle.metadata['source']
    if not manifest.get('data_is_synthetic'):
        raise ValueError('Known informative-region labels only exist for synthetic benchmark')
    ratios=[];corr=[]
    with torch.no_grad():
        for j in range(len(ds)):
            z=ds[j]
            x=z['x'][None];m=z['mask'][None];t=z['t'][None]
            _,phi,_,_,_,_=model(x,m,t,K=a.samples,mode='conditional')
            dt=t[:,1:]-t[:,:-1]
            derivative=(phi[:,:,1:]-phi[:,:,:-1])/dt[None]
            rel=(t-t[:,:1])/(t[:,-1:]-t[:,:1])
            mid=(rel[:,1:]+rel[:,:-1])*.5
            region=(mid>=.65)&(mid<=.75)
            if not region.any() or region.all():continue
            a1=derivative[:,:,region[0]].mean().item()
            a0=derivative[:,:,~region[0]].mean().item()
            ratios.append(a1/max(a0,1e-10))
            corr.append(a1-a0)
    report=dict(dataset='synthetic',n=len(ratios),
                mean_salient_vs_nonsalient_slope_ratio=float(np.mean(ratios)),
                mean_slope_difference=float(np.mean(corr)),
                claim='diagnostic only: temporal resolution density; no target error evaluated')
    Path(a.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

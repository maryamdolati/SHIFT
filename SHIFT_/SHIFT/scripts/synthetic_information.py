import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'krno'))
from shift_krno.model import SHIFT
from shift_krno.losses import mixture_nll


def task(n,seed,heterogeneous):
    rng=np.random.default_rng(seed);t=np.linspace(0,1,40,dtype=np.float32);q=np.linspace(1.05,1.5,10,dtype=np.float32)
    c=rng.uniform(.2,.8,n).astype(np.float32) if heterogeneous else np.full(n,.5,np.float32)
    width=.12 if heterogeneous else .48
    amp=rng.normal(size=n).astype(np.float32)
    region=np.abs(t[None]-c[:,None])<=width
    residual=rng.normal(0,1,(n,len(t))).astype(np.float32)
    signal=np.where(region,amp[:,None]+.1*residual,residual)
    marker=np.exp(-((t[None]-c[:,None])/width)**2).astype(np.float32)
    x=np.stack([signal,marker],-1).astype(np.float32)
    y=(amp[:,None]*np.cos(2*np.pi*(q[None]-1))+.05*rng.normal(size=(n,len(q)))).astype(np.float32)[...,None]
    return TensorDataset(*(torch.from_numpy(a) for a in (x,np.broadcast_to(t,(n,len(t))).copy(),np.broadcast_to(q,(n,len(q))).copy(),y,region)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--epochs',type=int,required=True);p.add_argument('--n-train',type=int,default=1024);p.add_argument('--n-val',type=int,default=256);p.add_argument('--n-test',type=int,default=256);p.add_argument('--lr',type=float,required=True);p.add_argument('--samples',type=int,default=4);a=p.parse_args()
    if a.samples<2 or a.epochs<1:raise ValueError('Require positive epochs and K>=2')
    records=[];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for heterogeneous in (False,True):
        for seed in a.seeds:
            datasets=[task(n,seed*10+i,heterogeneous) for i,n in enumerate((a.n_train,a.n_val,a.n_test))]
            for mode in ('fixed','deterministic','conditional','random','unconditional'):
                torch.manual_seed(seed);net=SHIFT(2,outputs=1).to(device);opt=torch.optim.AdamW(net.parameters(),lr=a.lr);best=float('inf');state=None
                loader=DataLoader(datasets[0],batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(seed))
                def calc(data):
                    net.eval();sq=mass=frac=n=0.
                    with torch.no_grad():
                        for x,t,q,y,region in DataLoader(data,batch_size=16):
                            x,t,q,y,region=(z.to(device) for z in (x,t,q,y,region));out=net(x,torch.ones_like(x),t,q,K=a.samples,mode=mode)
                            sq+=float((out.mean-y).square().sum());n+=y.numel();intervals=region[:,:-1]&region[:,1:];mass+=float((out.phi.diff(dim=-1)*intervals[None]).sum()/a.samples);frac+=float(intervals.float().mean(-1).sum())
                    return dict(mse=sq/n,informative_mass=mass/len(data),physical_interval_fraction=frac/len(data))
                for epoch in range(a.epochs):
                    net.train()
                    for x,t,q,y,_ in loader:
                        x,t,q,y=(z.to(device) for z in (x,t,q,y));out=net(x,torch.ones_like(x),t,q,K=a.samples,mode=mode);loss=mixture_nll(out,y);opt.zero_grad();loss.backward();opt.step()
                    value=calc(datasets[1])['mse']
                    if value<best:best=value;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                net.load_state_dict(state);records.append(dict(seed=seed,heterogeneous=heterogeneous,mode=mode,validation_mse=best,test=calc(datasets[2])))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(task='synthetic hypothesis test, not a KRNO benchmark',records=records),indent=2)+'\n')


if __name__=='__main__':main()

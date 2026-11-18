import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def pack(path,items,collate,record_ids):
    keys={'x':'observed_data','mask':'observed_mask','t':'observed_tp','q':'tp_to_predict','y':'data_to_predict','ymask':'mask_predicted_data'}
    buffers={k:[] for k in keys};xo=[0];yo=[0]
    for item in items:
        data=collate([item])
        vals={k:data[v][0].detach().cpu().numpy().astype(np.float32) for k,v in keys.items()}
        if len(vals['t'])<2 or not len(vals['q']):raise ValueError(f'Upstream example {item[0]} has unsupported empty/short support; no silent dropping')
        for k in keys:buffers[k].append(vals[k])
        xo.append(xo[-1]+len(vals['t']));yo.append(yo[-1]+len(vals['q']))
    np.savez_compressed(path,**{k:np.concatenate(v,axis=0) for k,v in buffers.items()},x_offsets=np.array(xo),y_offsets=np.array(yo),record_ids=np.asarray(record_ids,dtype=str))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--upstream-root',required=True)
    parser.add_argument('--dataset',required=True,choices=['mimic','ushcn','activity'])
    parser.add_argument('--out',required=True)
    a=parser.parse_args();root=Path(a.upstream_root).resolve();out=Path(a.out).resolve()/a.dataset
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Use a clean upstream source checkout')
    base=root/'experiments/scripts/irregular_time_series'
    sys.path.insert(0,str(base));os.chdir(base/'krno')
    from lib import parse_datasets as native
    from sklearn.model_selection import train_test_split
    device=torch.device('cpu')
    args=SimpleNamespace(dataset=a.dataset,history=3000 if a.dataset=='activity' else 24,device=device,n=int(1e8),n_months=48,pred_window=1 if a.dataset=='ushcn' else 1000 if a.dataset=='activity' else 24)
    if a.dataset=='mimic':raw=native.MIMIC('../data/mimic/',n_samples=args.n,device=device)
    elif a.dataset=='ushcn':raw=native.USHCN('../data/ushcn/',n_samples=args.n,device=device)
    else:raw=native.PersonActivity('../data/activity/',n_samples=args.n,download=True,device=device)
    seen,test=train_test_split(raw,train_size=.8,random_state=42,shuffle=True)
    train,val=train_test_split(seen,train_size=.75,random_state=42,shuffle=False)
    mn,mx,tm=native.get_data_min_max(seen,device)
    if a.dataset=='activity':tm=torch.tensor(args.history+args.pred_window)
    def collate(items):
        if a.dataset=='ushcn':return native.USHCN_variable_time_collate_fn(items,args,device,time_max=tm)
        return native.variable_time_collate_fn(items,args,device,data_min=mn,data_max=mx,time_max=tm)
    out.mkdir(parents=True,exist_ok=True)
    hashes={};split_ids={};source_hashes={}
    for name,records in [('train',train),('val',val),('test',test)]:
        split_ids[name]=[str(r[0]) for r in records]
        source_hashes[name]=hashlib.sha256(json.dumps(split_ids[name]).encode()).hexdigest()
        if a.dataset=='ushcn':items=native.USHCN_time_chunk(records,args,device);ids=[str(r[0]) for r in items]
        elif a.dataset=='activity':items=native.Activity_time_chunk(records,args,device);ids=[str(r[0]).rsplit('_',1)[0] for r in items]
        else:items=records;ids=[str(r[0]) for r in items]
        path=out/f'{name}.npz';pack(path,items,collate,ids);hashes[path.name]=digest(path)
    (out/'manifest.json').write_text(json.dumps(dict(source_repository='https://github.com/srinathdama/ShiftingTime',source_commit=commit,source_split_hashes=hashes,raw_record_id_hashes=source_hashes,raw_record_ids=split_ids,split_seed=42,protocol='krno_upstream_collate',preprocessing_description='Native KRNO dataset constructors, split, chunking and batch-size-one collate',history=args.history,pred_window=args.pred_window,normalization_min=mn.tolist(),normalization_max=mx.tolist(),time_max=tm.tolist()),indent=2))


if __name__=='__main__':main()

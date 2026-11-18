import argparse
import importlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import torch
from shift_krno.run import parser,load_dataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--upstream-root',required=True)
    a=p.parse_args()
    errors=[]
    upstream=Path(a.upstream_root)
    try:
        sha=subprocess.run(['git','rev-parse','HEAD'],cwd=upstream,check=True,
                           capture_output=True,text=True).stdout.strip()
    except Exception as exc:
        sha='UNVERIFIED';errors.append(f'Upstream git commit not verifiable: {exc}')
    try:
        mod=importlib.import_module('khatriraonop')
        khatri_present=True
    except Exception as exc:
        khatri_present=False;errors.append(f'Missing original khatriraonop: {exc}')
    try:
        cfg=parser().parse_args(['--dataset',a.dataset,'--data-root',a.data_root])
        data=load_dataset(cfg)
        contract=data.metadata
        if len(data.train)==0 or len(data.val)==0:
            errors.append('Train/validation empty: no valid model selection possible')
        if data.metadata.get('excluded',0):
            errors.append('M4 official test series excluded by lookback')
    except Exception as exc:
        contract=None;errors.append(f'Data contract failed: {exc}')
    report=dict(platform=platform.platform(),python=sys.version,
       torch=torch.__version__,upstream_commit=sha,
       original_krno_installed=khatri_present,dataset=a.dataset,
       data_contract=contract,errors=errors,passed=not errors)
    print(json.dumps(report,indent=2,default=str))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()

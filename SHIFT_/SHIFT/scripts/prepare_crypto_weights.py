import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--asset-ids',nargs=14,type=int,required=True);a=p.parse_args()
    if a.asset_ids!=list(range(14)):raise ValueError('Confirm official asset order 0..13 without automatic relabeling')
    spec=importlib.util.spec_from_file_location('native_metrics',a.upstream_root/'experiments/utils/KNF_eval_metrics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    weights=np.asarray(m.WRMSE.__defaults__[0],dtype=np.float32)
    if weights.shape!=(14,) or not np.isfinite(weights).all() or not (weights>0).all():raise ValueError('Unexpected upstream metric weights')
    a.out.mkdir(parents=True,exist_ok=True);np.save(a.out/'weights.npy',weights);(a.out/'assets.json').write_text(json.dumps(dict(asset_ids=a.asset_ids,source='official KRNO KNF_eval_metrics.py WRMSE defaults; asset identity must be independently checked against native data'),indent=2))


if __name__=='__main__':main()

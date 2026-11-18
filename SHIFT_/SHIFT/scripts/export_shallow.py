import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',required=True)
    p.add_argument('--test',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    def load(path,split):
        with np.load(path,allow_pickle=False) as z:
            return np.concatenate([z[f'U_{split}'][:,None],z[f's_{split}']],axis=1).astype(np.float32)
    target=out/'shallow_water.npz'
    np.savez(target,train=load(a.train,'train'),test=load(a.test,'test')[:,:60])
    (out/'manifest.json').write_text(json.dumps(dict(source_repository='LOCA',source_commit=a.source_commit,sha256=hash_file(target),raw_train_sha256=hash_file(Path(a.train)),raw_test_sha256=hash_file(Path(a.test)),protocol='original train/test, validation split after windowing'),indent=2))


if __name__=='__main__':main()

import argparse,json
from pathlib import Path


def check(a,b,comparison='operator_matched',max_param_ratio=1.1):
    problems=[]
    for field in ('dataset','seed','lookback','block','objective','epochs','batch_size',
                  'grad_accum','train_stride','test_stride','lr','weight_decay',
                  'integral_layers','probabilistic',
                  'train_samples','eval_samples','refit','refit_epochs'):
        if a['arguments'].get(field)!=b['arguments'].get(field):
            problems.append(f'Mismatched {field}: {a["arguments"].get(field)} != {b["arguments"].get(field)}')
    if comparison=='operator_matched':
        for name in ('width','kernel_hidden'):
            if a['arguments'].get(name)!=b['arguments'].get(name):
                problems.append(f'Mismatched operator {name}')
    if a['data_inventory']!=b['data_inventory']:
        problems.append('Different input file inventory hashes')
    if a['data_contract']!=b['data_contract']:
        problems.append('Different dataset/preprocessing/split contract')
    if a['arguments']['architecture']==b['arguments']['architecture']:
        problems.append('Not a pair of distinct reference and warp architectures')
    names={a['arguments']['architecture'],b['arguments']['architecture']}
    if names!={'krno_reference','krno_warp'}:
        problems.append('Only compare krno_reference and krno_warp as primary causal pair')
    if a['arguments'].get('reg_strength',0)!=0 or b['arguments'].get('reg_strength',0)!=0:
        problems.append('Warp regularization differs from baseline objective; rerun with --reg-strength 0')
    na=a['parameter_count']['trainable_params']
    nb=b['parameter_count']['trainable_params']
    ratio=max(na/nb,nb/na)
    if comparison=='capacity_matched' and ratio>max_param_ratio:
        problems.append(f'Unmatched trainable parameter counts {ratio:.3f} > {max_param_ratio}')
    return dict(passed=not problems, problems=problems, parameter_ratio=ratio,
                comparison=comparison,method='code and dataset contract equality, not replicated test accuracy')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shift',required=True)
    p.add_argument('--krno-control',required=True)
    p.add_argument('--comparison',choices=['operator_matched','capacity_matched'],default='operator_matched')
    p.add_argument('--max-param-ratio',type=float,default=1.1)
    a=p.parse_args()
    r=check(json.loads(Path(a.shift).read_text()),
            json.loads(Path(a.krno_control).read_text()),
            a.comparison,a.max_param_ratio)
    print(json.dumps(r,indent=2))
    if not r['passed']:raise SystemExit('FAIRNESS AUDIT FAILED')

if __name__=='__main__':main()

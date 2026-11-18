import copy
import torch


def donor_evaluate(model,bundle,args,evaluate):
    if len(bundle.val)==0:raise ValueError('An independent validation donor is required')
    d=bundle.val[0]
    x,m,t=(d[k][None] for k in ('x','mask','t'))
    if x.ndim==5:x=x.mean((2,3));m=m.amin((2,3))
    warp=model.warp
    warp.donor=(x,m,t)
    audit_args=copy.copy(args);audit_args.mode='shuffled'
    try:
        return dict(donor_source='first validation history, no target values',evaluation=evaluate(model,bundle,audit_args))
    finally:
        del warp.donor

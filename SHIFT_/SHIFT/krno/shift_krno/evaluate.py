from collections import defaultdict
import numpy as np
import torch
from .losses import (nmae, smape, crypto_blocks, trajectory_blocks,
                     shallow_relative_l2,interval_coverage_normal,uncertainty_error_corr)


def _device(model): return next(model.parameters()).device


def tensors(batch,device):
    return tuple(batch[k].to(device) for k in ('x','mask','t','q','y','ymask'))


@torch.no_grad()
def one_block(model,x,mask,t,q,K,mode):
    model.eval()
    return model(x,mask,t,q,K=K,mode=mode)


@torch.no_grad()
def temporal_rollout(model,history,horizon,block,lookback,K=8,mode='conditional',
                     observed_mask=None,origin_time=0.0,step=1.0,
                     audit_callback=None):


    if history.ndim!=3 or history.size(1)!=lookback:raise ValueError("Wrong history")
    device=history.device
    B,L,C=history.shape
    mask=observed_mask if observed_mask is not None else torch.ones_like(history)
    if not isinstance(origin_time,torch.Tensor):
        origin_time=torch.full((B,),float(origin_time),device=device)
    t=(origin_time[:,None]+(torch.arange(-lookback,0,device=device)*step))
    predictions=[]; temporal=[]; value=[]
    n=0
    while n < horizon:
        width=min(block,horizon-n)
        q=origin_time[:,None]+(n+torch.arange(block,device=device)+0.0)*step
        out=one_block(model,history,mask,t,q,K,mode)
        if width!=block:
            from dataclasses import replace
            out=replace(out,mean=out.mean[:,:width],variance=out.variance[:,:width],u_value=out.u_value[:,:width],u_temporal=out.u_temporal[:,:width],component_mean=out.component_mean[:,:,:width],component_variance=out.component_variance[:,:,:width])
        if audit_callback is not None:audit_callback(out,n,n+width)
        predictions.append(out.mean)
        temporal.append(out.u_temporal)
        value.append(out.u_value)
        history=torch.cat([history,out.mean],1)[:,-lookback:]
        mask=torch.cat([mask,torch.ones_like(out.mean)],1)[:,-lookback:]
        t=torch.cat([t,q],1)[:,-lookback:]
        n+=width
    return (torch.cat(predictions,1),torch.cat(value,1),torch.cat(temporal,1))


def _rescale(z,mu,sd):
    return torch.as_tensor(z)*sd+mu


@torch.no_grad()
def eval_darts(model,bundle,K=8,mode='conditional',probabilistic_audit=False):
    z=bundle.series
    meta=bundle.metadata
    L=meta['lookback'];P=meta['forecast_block'];o=meta['val_end'];h=meta['test_horizon']
    dev=_device(model)
    hist=torch.tensor(z[o-L:o],device=dev)[None]
    real=torch.tensor(z[o:o+h],device=dev)[None]
    mu,sd=meta['mean'],meta['std']
    from .uncertainty_audit import ProbabilisticAudit
    acc=ProbabilisticAudit() if probabilistic_audit else None
    if acc is not None and not getattr(model,'has_predictive_variance',False):
        raise ValueError('Probabilistic audit needs a learned conditional variance head')
    def audit(out,start,stop):
        acc.update(out,_rescale(real[:,start:stop],mu,sd),
                   mean_override=_rescale(out.component_mean,mu,sd),
                   var_override=out.component_variance*sd**2)
    pred,va,vt=temporal_rollout(model,hist,h,P,L,K,mode,
        origin_time=float(o),audit_callback=audit if acc is not None else None)
    result=dict(nmae=float(nmae(_rescale(real,mu,sd),_rescale(pred,mu,sd))),
                horizon=h,rolled_blocks=int(np.ceil(h/P)),
                temporal_var_mean=float(vt.mean()*sd*sd),
                value_var_mean=float(va.mean()*sd*sd))
    if acc is not None:result['probabilistic']=acc.result()
    return result


@torch.no_grad()
def eval_m4(model,bundle,K=8,mode='conditional',probabilistic_audit=False):
    meta=bundle.metadata;L=meta['lookback'];P=meta['pred_block'];h=meta['official_horizon']
    if meta['excluded']:
        raise ValueError('M4 official evaluation prohibits dropped test series: choose a shorter lookback')
    dev=_device(model)
    smapes=[]; values=[]
    from .uncertainty_audit import ProbabilisticAudit
    acc=ProbabilisticAudit() if probabilistic_audit else None
    if acc is not None and not getattr(model,'has_predictive_variance',False):
        raise ValueError('Probabilistic audit needs a learned conditional variance head')
    for j,series in enumerate(bundle.series):
        o=meta['stats'][j]['train_n']
        z=torch.as_tensor(series,device=dev)
        mu=meta['stats'][j]['mean'];sd=meta['stats'][j]['std']
        def audit(out,start,stop):
            acc.update(out,_rescale(z[o+start:o+stop][None],mu,sd),
                       mean_override=_rescale(out.component_mean,mu,sd),
                       var_override=out.component_variance*sd**2)
        pred,va,vt=temporal_rollout(model,z[o-L:o][None],h,P,L,K,mode,
            origin_time=float(o),audit_callback=audit if acc is not None else None)
        pred=_rescale(pred,mu,sd); truth=_rescale(z[o:o+h][None],mu,sd)
        smapes.append(float(smape(truth,pred)))
        values.append((float(va.mean()*sd*sd),float(vt.mean()*sd*sd)))
    result=dict(smape=float(np.mean(smapes)),n_series=len(smapes),
                horizon=h,rollout_blocks=int(np.ceil(h/P)),
                value_var=float(np.mean([a for a,b in values])),
                temporal_var=float(np.mean([b for a,b in values])))
    if acc is not None:result['probabilistic']=acc.result()
    return result


@torch.no_grad()
def eval_crypto(model,bundle,K=64,mode='conditional',batch_size=128,probabilistic_audit=False):
    from torch.utils.data import DataLoader
    dev=_device(model);ds=bundle.test
    if ds is None:raise ValueError('Crypto test windows unavailable')
    data=defaultdict(dict)
    norm=bundle.metadata
    from .uncertainty_audit import ProbabilisticAudit
    acc=ProbabilisticAudit() if probabilistic_audit else None
    if acc is not None and not getattr(model,'has_predictive_variance',False):
        raise ValueError('Probabilistic audit needs a learned conditional variance head')
    for batch in DataLoader(ds,batch_size=batch_size,shuffle=False):
        x,mask,t,q,y,ym=tensors(batch,dev)
        out=one_block(model,x,mask,t,q,K,mode)
        pred=out.mean
        if pred.shape!=y.shape:raise ValueError("Crypto selected evaluation requires full 15-step prediction")
        if acc is not None:
            asset_ids=batch['series'].tolist()
            mu=torch.as_tensor([norm['per_asset_mean'][i][-1] for i in asset_ids],
                               device=dev,dtype=y.dtype)[None,:,None,None]
            sd=torch.as_tensor([norm['per_asset_std'][i][-1] for i in asset_ids],
                               device=dev,dtype=y.dtype)[None,:,None,None]
            yc=y[...,-1:]*sd[0]+mu[0]
            acc.update(out,yc,
                       mean_override=out.component_mean[...,-1:]*sd+mu,
                       var_override=out.component_variance[...,-1:]*sd.square())
        for j in range(len(x)):
            asset=int(batch['series'][j]);origin=int(batch['origin'][j]);
            mu=torch.as_tensor(norm['per_asset_mean'][asset],device=dev)
            sd=torch.as_tensor(norm['per_asset_std'][asset],device=dev)

            yy=y[j]*sd+mu;pp=pred[j]*sd+mu
            if asset in data[origin]:raise ValueError('Duplicate asset/origin')
            data[origin][asset]=(yy.cpu(),pp.cpu())
    true=[];pred=[]
    for o in sorted(data):
        if set(data[o])!=set(range(14)):
            raise ValueError(f'Missing assets at crypto origin {o}: {set(data[o])}')
        true.append(torch.stack([data[o][a][0] for a in range(14)]))
        pred.append(torch.stack([data[o][a][1] for a in range(14)]))
    if not true:raise ValueError('No crypto test origins')
    y=torch.stack(true);p=torch.stack(pred)
    w=torch.tensor(norm['weights'])
    result=crypto_blocks(y,p,w)
    result.update(n_origins=len(true),n_assets=14,feature='last channel only',
                  metric_source='Official KRNO KNF_eval_metrics.py (squared weights)',
                  reporting_scale='Multiply original-units WRMSE by 1000 for paper table',
                  metric_caveat='Official total covers all 15 horizons; official long scores step 11 only')
    if acc is not None:result['probabilistic']=acc.result()
    return result


@torch.no_grad()
def eval_trajectory(model,bundle,K=8,mode='conditional',probabilistic_audit=False):
    dev=_device(model)
    series,origins=bundle.series
    L=bundle.metadata['lookback'] if 'lookback' in bundle.metadata else 30
    P=bundle.metadata['forecast_block']
    trues=[];preds=[]
    from .uncertainty_audit import ProbabilisticAudit
    acc=ProbabilisticAudit() if probabilistic_audit else None
    if acc is not None and not getattr(model,'has_predictive_variance',False):
        raise ValueError('Probabilistic audit needs a learned conditional variance head')
    for i,o in origins:
        data=torch.as_tensor(series[i],device=dev)
        stats=bundle.metadata['stats'][i]
        mu=torch.as_tensor(stats['mean'],device=dev);sd=torch.as_tensor(stats['std'],device=dev)
        def audit(out,start,stop):
            acc.update(out,data[o+start:o+stop][None]*sd+mu,mean_override=out.component_mean*sd+mu,var_override=out.component_variance*sd.square())
        pr,_,_=temporal_rollout(model,data[o-L:o][None],30,P,L,K,mode,
            origin_time=float(o),audit_callback=audit if acc is not None else None)
        trues.append((data[o:o+30]*sd+mu).cpu());preds.append((pr[0]*sd+mu).cpu())
    if len(origins)!=5*bundle.metadata['test_examples']:
        raise ValueError('Not evaluating all official trajectory origins and trajectories')
    result=trajectory_blocks(torch.stack(trues),torch.stack(preds))
    result.update(n_test_windows=len(origins),horizon=30,
                  observation_source='Test history observed at each origin; no future within 30-step rollout')
    if acc is not None:result['probabilistic']=acc.result()
    return result


@torch.no_grad()
def eval_irregular(model,bundle,K=8,mode='conditional',batch_size=1,probabilistic_audit=False):
    from torch.utils.data import DataLoader
    from .losses import variable_error_sums,macro_from_sums
    dev=_device(model);sse=0.;sae=0.;n=0
    macro=bundle.metadata.get("dataset") in ("mimic","ushcn","activity")
    if bundle.metadata.get("variable_length"):batch_size=1
    from .uncertainty_audit import ProbabilisticAudit
    acc=ProbabilisticAudit() if probabilistic_audit else None
    if acc is not None and not getattr(model,'has_predictive_variance',False):
        raise ValueError('Probabilistic audit requires a trained conditional variance head')
    for batch in DataLoader(bundle.test,batch_size=batch_size,shuffle=False):
        x,m,t,q,y,ym=tensors(batch,dev)
        out=one_block(model,x,m,t,q,K,mode)
        se,ae,count=variable_error_sums(out.mean,y,ym)
        sse=sse+se.double().cpu();sae=sae+ae.double().cpu();n=n+count.double().cpu()
        if acc is not None:acc.update(out,y,ym)
    if n.sum()==0:raise ValueError('No observed irregular test targets')
    result=dict(mse=float(macro_from_sums(sse,n) if macro else sse.sum()/n.sum()),mae=float(macro_from_sums(sae,n) if macro else sae.sum()/n.sum()),masked_count=int(n.sum()),aggregation='variable-wise' if macro else 'observation-wise')
    if acc is not None:result['probabilistic']=acc.result()
    return result


@torch.no_grad()
def eval_shallow(model,bundle,K=8,mode='conditional',probabilistic_audit=False):


    dev=_device(model);error_sum=torch.zeros(3,dtype=torch.float64);n_trajectories=0
    from .uncertainty_audit import ProbabilisticAudit
    acc=ProbabilisticAudit() if probabilistic_audit else None
    if acc is not None and not getattr(model,'has_predictive_variance',False):
        raise ValueError('Probabilistic audit needs a learned conditional variance head')
    for arr in bundle.series:
        field=torch.as_tensor(arr,device=dev)
        if field.shape[0]<60:
            raise ValueError('Official Shallow Water needs >=60 snapshots')
        history=field[:5].unsqueeze(0)
        t=torch.arange(5,device=dev).float()[None]*.01
        blocks=[history]
        calls=0
        for origin in range(5,60,5):
            q=torch.arange(origin,origin+5,device=dev).float()[None]*.01
            out=one_block(model,history,torch.ones_like(history),t,q,K,mode)
            if acc is not None:
                acc.update(out,field[origin:origin+5][None])
            blocks.append(out.mean)
            history=torch.cat([history,out.mean],1)[:,-5:]
            t=torch.cat([t,q],1)[:,-5:]
            calls+=1
        whole=torch.cat(blocks,dim=1)
        if whole.size(1)!=60 or calls!=11:
            raise AssertionError('Expected precisely 60 total frames and 11 model calls')
        error_sum+=shallow_relative_l2(field[:60][None],whole,observed_steps=5).double().cpu()
        n_trajectories+=1


    if not n_trajectories:raise ValueError("No shallow test trajectories")
    errors=error_sum/n_trajectories
    result=dict(relative_l2_rho=float(errors[0]),relative_l2_u=float(errors[1]),
                relative_l2_v=float(errors[2]),total_steps=60,observed_steps=5,
                scored_steps=55,recursive_steps=55,model_calls_per_trajectory=11,
                n_test_trajectories=len(bundle.series),
                scoring_window='Only 55 future frames; excludes 5 observed context frames')
    if acc is not None:result['probabilistic']=acc.result()
    return result

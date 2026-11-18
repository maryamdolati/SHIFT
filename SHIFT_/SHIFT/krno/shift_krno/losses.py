import math
import torch
import torch.nn.functional as F


def masked_mean(values, valid=None):
    if valid is None: return values.mean()
    v=valid.to(values.dtype).expand_as(values)
    return (values*v).sum()/v.sum().clamp_min(1)


def warp_regularizer(out, smooth=2e-3, rate=1e-5, kl=0.0,
                     length=5e-3, shape=False):
    increments=out.raw_increments
    pre_span=increments.sum(-1)
    d=increments[:,:,1:]-increments[:,:,:-1]
    loss=smooth*d.square().mean() if d.numel() else increments.sum()*0
    if shape:
        loss=loss+rate*(out.rates.clamp_min(1e-8).log().square().mean())
    else:
        loss=loss-rate*out.rates.clamp_min(1e-8).log().mean()
    if length:
        loss=loss+length*(pre_span-out.interval_widths.sum(-1)[None]).square().mean()
    if kl:
        v=out.logvar.exp()
        loss=loss+kl*0.5*(out.mu.square()+v-out.logvar-1).sum(-1).mean()
    return loss


def mixture_nll(out, y, valid=None):


    mu=out.component_mean
    var=out.component_variance.clamp_min(1e-8)
    logprob=-0.5*((y[None]-mu).square()/var+var.log()+math.log(2*math.pi))
    if valid is not None: logprob=logprob*valid[None]
    per_sample=logprob.flatten(2).sum(-1)
    observed=(torch.ones_like(y) if valid is None else valid).flatten(1).sum(-1).clamp_min(1)
    return ((-torch.logsumexp(per_sample,dim=0)+math.log(mu.size(0)))/observed).mean()


def component_nll(out,y,valid=None):
    mu,v=out.component_mean,out.component_variance.clamp_min(1e-8)
    loss=0.5*((y[None]-mu).square()/v+v.log()+math.log(2*math.pi))
    return masked_mean(loss,valid[None] if valid is not None else None)


def forecast_loss(out,y,valid=None,objective="mse",rate=0.0,
                  reg_kwargs=None,macro=False):
    point=masked_macro_mse(out.mean,y,valid) if macro else masked_mean((out.mean-y).square(),valid)
    if objective=="mse":
        loss=point
    elif objective=="mix_nll":
        loss=mixture_nll(out,y,valid)
    elif objective=="component_nll":
        loss=component_nll(out,y,valid)
    elif objective=="mse_plus_nll":
        loss=point+rate*component_nll(out,y,valid)
    else: raise ValueError(objective)
    if reg_kwargs: loss=loss+warp_regularizer(out,**reg_kwargs)
    return loss


def smape(truth,pred):
    denom=truth.abs()+pred.abs()
    return (200*(truth-pred).abs()/denom.clamp_min(1e-8)).masked_fill(denom<1e-8,0).mean()


def nmae(truth,pred):
    return (truth-pred).abs().mean()/truth.abs().mean().clamp_min(1e-8)


def wrmse_crypto(y, pred, weights):


    if weights is None:
        raise ValueError('No Crypto weights: cannot compute official WRMSE')
    if y.shape != pred.shape or y.ndim != 4 or y.size(-1) != 8:
        raise ValueError('Expected matching [origin, asset, horizon, 8] tensors')
    w = torch.as_tensor(weights, dtype=y.dtype, device=y.device)
    if w.ndim != 1 or w.numel() != y.size(1) or not torch.isfinite(w).all() or not (w > 0).all():
        raise ValueError('Weights must be finite positive values in official asset order')
    residual = pred[..., -1] - y[..., -1]
    return ((residual * w[None, :, None]).square().mean()).sqrt()


def crypto_blocks(y,pred,weights):
    if y.shape!=pred.shape or y.ndim!=4 or y.shape[-2:]!=(15,8):raise ValueError('Expected [origin,asset,15,8]')
    def score(a,b):return wrmse_crypto(y[:,:,a:b],pred[:,:,a:b],weights).item()
    total=score(0,15)
    return {'1-5':score(0,5),'6-10':score(5,10),'11-15':score(10,15),'official_short':score(0,5),'official_medium':score(5,10),'official_long_step':score(10,11),'total':total,'total_all_steps':total,'paper_described_total_all_15_steps':total,'official_total_is_final_horizon_only':False}


def trajectory_blocks(y,pred):
    if y.shape != pred.shape or y.shape[-2:]!=(30,2):
        raise ValueError("Expected trajectory [...,30,2]")
    return {f"{a+1}-{b}":(y[...,a:b,:]-pred[...,a:b,:]).square().mean().sqrt().item()
            for a,b in ((0,10),(10,20),(20,30))} | {"total":(y-pred).square().mean().sqrt().item()}


def shallow_relative_l2(y, pred, observed_steps=0):


    if y.shape != pred.shape or y.ndim != 5 or y.size(-1) != 3:
        raise ValueError('Expected matching [batch,time,32,32,3]')
    if not 0 <= observed_steps < y.size(1):
        raise ValueError('observed_steps must be smaller than target length')
    target = y[:, observed_steps:]
    forecast = pred[:, observed_steps:]
    axes = (1, 2, 3)
    numerator = (target - forecast).square().sum(dim=axes).sqrt()
    denominator = target.square().sum(dim=axes).sqrt().clamp_min(1e-12)
    return (numerator / denominator).mean(0)


def interval_coverage_normal(out,y,z=1.95996398454):
    l=out.mean-z*out.variance.clamp_min(1e-8).sqrt()
    u=out.mean+z*out.variance.clamp_min(1e-8).sqrt()
    return ((y>=l)&(y<=u)).float().mean()


def uncertainty_error_corr(out,y):

    x=out.u_temporal.flatten()
    e=(y-out.mean).square().flatten()
    x=x-x.mean(); e=e-e.mean()
    return (x*e).mean()/(x.square().mean().sqrt()*e.square().mean().sqrt()).clamp_min(1e-12)


def variable_error_sums(pred,target,mask=None):
    if pred.shape!=target.shape:raise ValueError('Prediction and target shapes differ')
    mask=torch.ones_like(target) if mask is None else mask.expand_as(target).to(target)
    err=torch.where(mask.bool(),pred-target,torch.zeros_like(target)).reshape(-1,target.shape[-1])
    n=mask.reshape(-1,target.shape[-1]).sum(0)
    return err.square().sum(0),err.abs().sum(0),n


def macro_from_sums(sums,counts):
    valid=counts>0
    if not valid.any():raise ValueError('No observed targets')
    return (sums[valid]/counts[valid]).mean()


def masked_macro_mse(pred,target,mask=None):
    se,_,n=variable_error_sums(pred,target,mask)
    return macro_from_sums(se,n)


def quadrature_mse(pred,target,L):
    return (pred-target).square().mean()*target.shape[1]/L


def shallow_field_losses(pred,target):
    if pred.shape!=target.shape or target.ndim!=5 or target.shape[-1]!=3:raise ValueError('Expected shallow field batches')
    from khatriraonop import quadrature
    grid=quadrature.get_quad_grid([quadrature.midpoint_vecs,quadrature.trapezoidal_vecs,quadrature.trapezoidal_vecs],list(target.shape[1:4]),[-1.]*3,[1.]*3)
    grid=quadrature.quad_grid_to_device(grid,target.device)
    grid=quadrature.quad_to_cartesian_grid(grid)
    return quadrature.integrate_on_domain((pred-target).square(),grid).reshape(-1,3).mean(0)


def manuscript_objective(out,y,mask,dataset,lookback=None,physical_pred=None,physical_target=None,nll_warmup=None,tail_steps=None,variance_floor=None,diversity_margin=None,reg_kwargs=None):
    p=out.mean
    if dataset in ('mimic','ushcn','activity'):
        loss=masked_macro_mse(p,y,mask)
        if dataset=='ushcn':
            if nll_warmup is None:raise ValueError('Original USHCN NLL warm-up schedule is required')
            loss=loss+3e-5*nll_warmup*component_nll(out,y,mask)
    elif dataset=='mujoco':
        if variance_floor is None:raise ValueError('Original MuJoCo uncertainty-floor threshold is required')
        loss=masked_mean((p-y).square(),mask)+.01*component_nll(out,y,mask)
        loss=loss+5*masked_mean((variance_floor-out.u_temporal).clamp_min(0),mask)+.02*masked_mean((out.variance-(p-y).square().detach()).square(),mask)
    elif dataset=='crypto':
        if tail_steps is None or not 1<=tail_steps<=y.shape[1]:raise ValueError('Original Crypto tail length is required')
        loss=.88*masked_mean((p-y).square(),mask)+.10*component_nll(out,y,mask)+.02*masked_mean((p[:,-tail_steps:]-y[:,-tail_steps:]).square(),mask[:,-tail_steps:] if mask is not None else None)
    elif dataset=='shallow':
        fields=shallow_field_losses(p,y)
        loss=(fields*fields.new_tensor([5,50,50])).sum()
    elif dataset.startswith('m4/') or dataset=='trajectory':
        loss=quadrature_mse(p,y,lookback)
        if dataset in ('m4/Quarterly','m4/Daily'):
            if physical_pred is None or physical_target is None:raise ValueError('M4 sMAPE objective needs explicit units')
            sm=smape(physical_target,physical_pred)
            loss=.7*loss+.3*sm if dataset=='m4/Quarterly' else sm/100+.1*loss
    else:raise ValueError(dataset)
    if reg_kwargs:loss=loss+warp_regularizer(out,**reg_kwargs)
    if diversity_margin is not None:
        loss=loss+diversity_margin[0]*(diversity_margin[1]-out.phi.var(dim=0,unbiased=False).mean()).clamp_min(0)
    return loss

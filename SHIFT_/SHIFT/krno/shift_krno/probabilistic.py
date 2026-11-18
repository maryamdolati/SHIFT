import math
import torch


def marginal_mixture_nll(component_mean,component_variance,target,mask=None):

    K=component_mean.size(0)
    var=component_variance.clamp_min(1e-10)
    logp=-.5*((target[None]-component_mean).square()/var+var.log()+math.log(2*math.pi))
    nll=-(torch.logsumexp(logp,dim=0)-math.log(K))
    if mask is None:return nll.mean()
    mask=mask.broadcast_to(target.shape).bool()
    if not mask.any():raise ValueError("No observed targets")
    return nll[mask].mean()


def _phi(t):return torch.exp(-.5*t*t)/math.sqrt(2*math.pi)
def _Phi(t):return .5*(1+torch.erf(t/math.sqrt(2)))

def _normal_absdiff(mu1,s1,mu2,s2):
    sd=(s1.square()+s2.square()).clamp_min(1e-12).sqrt()
    d=(mu1-mu2)/sd
    return 2*sd*_phi(d)+(mu1-mu2)*(2*_Phi(d)-1)


def gaussian_mixture_crps(means,variances,y):


    K=means.size(0)
    s=variances.clamp_min(1e-12).sqrt()
    delta=y[None]
    term1=_normal_absdiff(means,s,delta,torch.zeros_like(delta)).mean(0)
    term2=0
    for k in range(K):
        for j in range(K):
            term2=term2+_normal_absdiff(means[k],s[k],means[j],s[j])
    return (term1-.5*term2/(K*K)).mean()


def predictive_pit(means,variances,y):
    sig=variances.clamp_min(1e-10).sqrt()
    return _Phi((y[None]-means)/sig).mean(0)


def interval_quantile(means,variances,alpha=.025,steps=55):

    if not 0<alpha<1:raise ValueError(alpha)
    s=variances.clamp_min(1e-10).sqrt()
    lo=(means-9*s).min(0).values
    hi=(means+9*s).max(0).values
    for _ in range(steps):
        mid=.5*(lo+hi)
        cdf=_Phi((mid[None]-means)/s).mean(0)
        lo=torch.where(cdf<alpha,mid,lo)
        hi=torch.where(cdf>=alpha,mid,hi)
    return .5*(lo+hi)


def calibration_summary(out,y):
    means,vars_=out.component_mean,out.component_variance
    low=interval_quantile(means,vars_,.025)
    high=interval_quantile(means,vars_,.975)
    pit=predictive_pit(means,vars_,y)
    return dict(mixture_nll=float(marginal_mixture_nll(means,vars_,y)),
                mixture_crps=float(gaussian_mixture_crps(means,vars_,y)),
                coverage95=float(((y>=low)&(y<=high)).float().mean()),
                pit_mean=float(pit.mean()),
                temporal_var=float(out.u_temporal.mean()),
                conditional_value_var=float(out.u_value.mean()))

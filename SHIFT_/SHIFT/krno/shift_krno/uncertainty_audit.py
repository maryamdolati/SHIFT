import math
import torch
from .probabilistic import (
    marginal_mixture_nll, gaussian_mixture_crps,
    interval_quantile, predictive_pit,
)


class ProbabilisticAudit:
    def __init__(self):
        self.n=0
        self.examples=[]
        self.totals={k:0. for k in ('nll','crps','coverage95','interval_score95',
                                     'pit','u_temporal','u_value','calibration_error')}
        self.sums={k:0. for k in ('e','u','ee','uu','eu')}
        self.coverage_at={q:0. for q in (0.5,0.8,0.95)}

    @torch.no_grad()
    def update(self,out,y,valid=None,mean_override=None,var_override=None):
        m=mean_override if mean_override is not None else out.component_mean
        v=var_override if var_override is not None else out.component_variance
        if m.ndim!=y.ndim+1 or m.shape[1:]!=y.shape or v.shape!=m.shape:
            raise ValueError('Component predictions and target shapes do not match')
        if valid is None:valid=torch.ones_like(y)
        mask=valid.bool()
        n=int(mask.sum())
        if n==0:return
        per_mean=m.mean(0)
        per_temp=(m-per_mean).square().mean(0)
        for j in range(y.shape[0]):
            take=mask[j]
            if take.any():self.examples.append((float((per_mean[j]-y[j]).square()[take].mean()),float(per_temp[j][take].mean())))
        m=m[:,mask];v=v[:,mask].clamp_min(1e-10)
        yy=y[mask]

        nll=marginal_mixture_nll(m,v,yy)
        crps=gaussian_mixture_crps(m,v,yy)
        lo=interval_quantile(m,v,0.025)
        hi=interval_quantile(m,v,0.975)
        inside=((yy>=lo)&(yy<=hi)).float()
        alpha=.05
        score=(hi-lo)+(2/alpha)*(lo-yy).clamp_min(0)+(2/alpha)*(yy-hi).clamp_min(0)
        pit=predictive_pit(m,v,yy)
        um=m.mean(0)
        ut=(m-um).square().mean(0)
        uv=v.mean(0)
        err=(yy-um).square()
        self.n+=n
        for name,scalar in (('nll',nll),('crps',crps),('coverage95',inside.mean()),
                            ('interval_score95',score.mean()),('pit',pit.mean()),
                            ('u_temporal',ut.mean()),('u_value',uv.mean())):
            self.totals[name]+=float(scalar)*n
        for q in self.coverage_at:
            l=interval_quantile(m,v,(1-q)/2)
            h=interval_quantile(m,v,(1+q)/2)
            self.coverage_at[q]+=int(((yy>=l)&(yy<=h)).sum())
        self.sums['e']+=float(err.sum())
        self.sums['u']+=float(ut.sum())
        self.sums['ee']+=float((err*err).sum())
        self.sums['uu']+=float((ut*ut).sum())
        self.sums['eu']+=float((err*ut).sum())

    def result(self):
        if not self.n:return dict(observed=0,status='no observed test labels')
        n=self.n
        a={k:v/n for k,v in self.totals.items() if k!='calibration_error'}
        empirical={str(int(q*100)):self.coverage_at[q]/n for q in self.coverage_at}
        a['calibration_error']=sum(abs(empirical[str(int(q*100))]-q) for q in self.coverage_at)/len(self.coverage_at)
        cov=self.sums['eu']/n-(self.sums['e']/n)*(self.sums['u']/n)
        vr1=self.sums['ee']/n-(self.sums['e']/n)**2
        vr2=self.sums['uu']/n-(self.sums['u']/n)**2
        a['temporal_error_pearson']=cov/math.sqrt(vr1*vr2) if vr1>1e-20 and vr2>1e-20 else None
        if len(self.examples)>=4:
            import numpy as np
            from scipy.stats import spearmanr
            pairs=np.asarray(self.examples)
            order=np.argsort(pairs[:,0],kind='stable');k=len(order)//4
            low=float(pairs[order[:k],1].mean());high=float(pairs[order[-k:],1].mean())
            r,p=spearmanr(pairs[:,0],pairs[:,1]) if np.ptp(pairs[:,0])>0 and np.ptp(pairs[:,1])>0 else (float('nan'),float('nan'))
            a.update(sample_level_spearman=float(r) if np.isfinite(r) else None,sample_level_spearman_p=float(p) if np.isfinite(p) else None,temporal_variance_high_error_quartile=high,temporal_variance_low_error_quartile=low,high_to_low_quartile_ratio=high/low if low>0 else None,n_examples=len(pairs),example_unit='one update batch item; recursive blocks are distinct examples')
        return dict(scalar_observations=n,**a,coverage_levels=empirical,
                    mixture_type='scalar marginal Gaussian mixture',
                    uncertainty_claim='sensitivity to learned stochastic warp; not clock-identifiability')

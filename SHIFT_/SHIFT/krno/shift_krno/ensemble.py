from dataclasses import replace
import torch
from torch import nn


class FixedCoordinateEnsemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        if len(members)<2 or any(not getattr(m,'has_predictive_variance',False) for m in members):
            raise ValueError('At least two independently fitted members with trained variance heads are required')
        self.members=nn.ModuleList(members)
        self.has_predictive_variance=True

    def forward(self,x,mask,t,q,K=1,mode='fixed'):
        if mode!='fixed':raise ValueError('This control is a fixed-coordinate ensemble')
        forecasts=[m(x,mask,t,q,K=1,mode='fixed') for m in self.members]
        means=torch.cat([f.component_mean for f in forecasts],0)
        variances=torch.cat([f.component_variance for f in forecasts],0)
        mean=means.mean(0); value=variances.mean(0); between=(means-mean).square().mean(0)
        return replace(forecasts[0],mean=mean,variance=value+between,u_value=value,u_temporal=between,component_mean=means,component_variance=variances)

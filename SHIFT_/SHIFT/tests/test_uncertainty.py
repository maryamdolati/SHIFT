import torch
from shift_krno.probabilistic import (gaussian_mixture_crps,
    predictive_pit,interval_quantile,marginal_mixture_nll)


def test_probabilistic_metrics_finite_and_symmetric():
    m=torch.zeros(5,2,4,3)
    var=torch.ones_like(m)
    y=torch.zeros(2,4,3)
    assert abs(float(predictive_pit(m,var,y).mean())-.5)<1e-6
    assert abs(float(interval_quantile(m,var,.975).mean())-1.95996398)<1e-5
    assert float(gaussian_mixture_crps(m,var,y))>0
    assert float(marginal_mixture_nll(m,var,y))>0

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ShiftOutput:
    mean: torch.Tensor
    variance: torch.Tensor
    u_value: torch.Tensor
    u_temporal: torch.Tensor
    component_mean: torch.Tensor
    component_variance: torch.Tensor
    phi: torch.Tensor
    rates: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    raw_increments: torch.Tensor
    interval_widths: torch.Tensor


def validate_input(x, mask, times, future):
    if x.ndim != 3 or mask.shape != x.shape or times.shape != x.shape[:2]:
        raise ValueError("Require x/mask [B,T,C], times [B,T]")
    if x.size(1) < 2:
        raise ValueError("At least two observation support points required")
    if not torch.isfinite(x).all() or not torch.isfinite(times).all() or not torch.isfinite(future).all():
        raise ValueError("Inputs contain NaN/inf; use benchmark placeholders plus masks")
    if not torch.all(times[:, 1:] > times[:, :-1]):
        raise ValueError("Processed timestamps must be strictly increasing")
    if future.ndim != 2 or future.shape[0] != x.shape[0]:
        raise ValueError("Future times require shape [B,H]")
    if not torch.all(future[:, 0] > times[:, -1]):
        raise ValueError("Future must be after the last observed time")
    if not torch.all(future[:, 1:] > future[:, :-1]):
        raise ValueError("Future timestamps must strictly increase")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Masks must be 0 or 1")


def latent_resample(z, phi, times, length=None):


    B, T, C = z.shape
    M = length or T
    u = torch.linspace(0, 1, M, device=z.device, dtype=z.dtype)
    u_exp = u[None, :].expand(B, -1).contiguous()

    idx = torch.searchsorted(phi.contiguous(), u_exp.contiguous(), right=True) - 1
    idx = idx.clamp(0, T - 2)
    ind = idx.unsqueeze(-1).expand(-1, -1, C)
    zl = z.gather(1, ind)
    zr = z.gather(1, ind + 1)
    pl = phi.gather(1, idx)
    pr = phi.gather(1, idx + 1)
    lam = (u_exp - pl) / (pr - pl).clamp_min(1e-8)
    interp = zl + lam.unsqueeze(-1) * (zr - zl)
    ds = times[:, 1:] - times[:, :-1]
    slopes = ((phi[:, 1:] - phi[:, :-1]) / ds).gather(1, idx)
    return interp, slopes.unsqueeze(-1)


def original_uneven_trapezoid(x):


    if x.ndim!=1 or x.numel()<2 or not torch.all(x[1:]>x[:-1]):
        raise ValueError('Increasing 1D quadrature nodes required')
    w=torch.empty_like(x)
    w[0]=.5*(x[1]-x[0])
    w[-1]=.5*(x[-1]-x[-2])
    w[1:-1]=x[2:]-x[1:-1]
    return w,x


class TemporalEncoder(nn.Module):
    def __init__(self, channels, hidden=32):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(2 * channels + 2, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden))

    def forward(self, x, mask, times):
        dt = torch.cat([torch.zeros_like(times[:, :1]), times[:, 1:] - times[:, :-1]], 1)
        rel = (times - times[:, :1]) / (times[:, -1:] - times[:, :1]).clamp_min(1e-8)
        feats = torch.cat([x * mask, mask, dt.unsqueeze(-1), rel.unsqueeze(-1)], -1)
        return self.lin(feats)


class ConditionalWarp(nn.Module):
    def __init__(self, channels, hidden=32, warp_hidden=64, rate_floor=1e-6):
        super().__init__()
        self.rate_floor=rate_floor
        self.encoder = TemporalEncoder(channels, hidden)
        ctx = 2 * hidden + 2 * channels + 1
        self.mean_net = nn.Sequential(nn.Linear(ctx, warp_hidden), nn.GELU(),
                                      nn.Linear(warp_hidden, 1))
        self.var_net = nn.Sequential(nn.Linear(ctx, warp_hidden), nn.GELU(),
                                     nn.Linear(warp_hidden, 1))
        self.global_mu = nn.Parameter(torch.zeros(()))
        self.global_lv = nn.Parameter(torch.full((), -2.0))

    def forward(self, x, mask, times, K=4, mode="conditional", external_noise=None):
        z = self.encoder(x, mask, times)
        ds = times[:, 1:] - times[:, :-1]
        ctx = torch.cat([z[:, :-1], z[:, 1:], mask[:, :-1], mask[:, 1:],
                         ds.unsqueeze(-1)], -1)
        mu = self.mean_net(ctx).squeeze(-1)
        lv = self.var_net(ctx).squeeze(-1).clamp(-12, 6)
        if mode in ("unconditional", "random"):
            mu = self.global_mu.expand_as(mu) if mode == "unconditional" else torch.zeros_like(mu)
            lv = self.global_lv.expand_as(lv) if mode == "unconditional" else torch.zeros_like(lv)
        if mode == "shuffled" and x.size(0) < 2 and not hasattr(self,"donor"):
            raise ValueError("Shuffled-warp ablation requires an independent donor or >=2 examples")
        if mode == "fixed":
            phi = (times - times[:, :1]) / (times[:, -1:] - times[:, :1])
            rates = torch.ones((K, *ds.shape), device=ds.device, dtype=ds.dtype)
            return z, phi[None].expand(K, -1, -1), rates, mu, lv, ds
        if mode == "deterministic":
            a = mu.unsqueeze(0).expand(K, -1, -1)
        elif mode in ("conditional", "unconditional", "random", "shuffled"):
            eps = (torch.randn((K, *mu.shape), device=mu.device, dtype=mu.dtype)
                   if external_noise is None else external_noise)
            if eps.shape != (K, *mu.shape):
                raise ValueError("Wrong external-noise shape")
            a = mu.unsqueeze(0) + (lv.exp() + 1e-6).sqrt().unsqueeze(0) * eps
        else:
            raise ValueError(f"Unrecognized warp mode: {mode}")
        rates = F.softplus(a).clamp_min(self.rate_floor)
        increments = rates * ds.unsqueeze(0)
        phi = torch.cat([torch.zeros_like(increments[:, :, :1]),
                         increments.cumsum(-1)], -1)
        phi = phi / phi[:, :, -1:]
        if not torch.all(phi[:,:,1:]>phi[:,:,:-1]):raise FloatingPointError("Non-increasing normalized temporal map")
        if mode == "shuffled":


            if hasattr(self,"donor"):
                dx,dm,dt=self.donor
                dx,dm,dt=dx.to(x),dm.to(mask),dt.to(times)
                _,dp,_,_,_,_=self(dx,dm,dt,K=K,mode="conditional")
                phi=transfer_donor_map(dp,dt,times)
            else:
                phi=transfer_donor_map(phi.roll(1,1),times.roll(1,0),times)
            rates = (phi[:, :, 1:] - phi[:, :, :-1]) / ds.unsqueeze(0)
        return z, phi, rates, mu, lv, ds


def transfer_donor_map(phi, donor_times, recipient_times):
    donor=(donor_times-donor_times[:,:1])/(donor_times[:,-1:]-donor_times[:,:1])
    recipient=(recipient_times-recipient_times[:,:1])/(recipient_times[:,-1:]-recipient_times[:,:1])
    if donor.size(0)==1 and recipient.size(0)>1:
        donor=donor.expand(recipient.size(0),-1)
        phi=phi.expand(-1,recipient.size(0),-1)
    if donor.size(0)!=recipient.size(0):raise ValueError("Donor and recipient batch mismatch")
    idx=(torch.searchsorted(donor.contiguous(),recipient.contiguous(),right=True)-1).clamp(0,donor.size(1)-2)
    left=donor.gather(1,idx);right=donor.gather(1,idx+1)
    w=(recipient-left)/(right-left).clamp_min(1e-8)
    kidx=idx[None].expand(phi.size(0),-1,-1)
    result=phi.gather(2,kidx)+w[None]*(phi.gather(2,kidx+1)-phi.gather(2,kidx))
    if not (result[:,:,1:]>result[:,:,:-1]).all():raise FloatingPointError("Donor map must be strictly monotone")
    return result


class MultiscaleSpectral(nn.Module):
    def __init__(self, channels, max_fft_bins=64):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, channels, kernel_size=k, padding=k//2),
                          nn.GELU(), nn.Conv1d(channels, channels, 1)) for k in (3, 5, 9)
        ])
        self.gate = nn.Linear(channels + 1, 3)
        self.freq_real = nn.Parameter(torch.ones(channels, max_fft_bins))
        self.freq_imag = nn.Parameter(torch.zeros(channels, max_fft_bins))
        self.spec_proj = nn.Conv1d(channels, channels, 1)
        self.fuse_proj = nn.Conv1d(channels, channels, 1)
        self.beta_logit = nn.Parameter(torch.tensor(-2.4))

    def forward(self, z, jac, strength=1.0):
        base = z.transpose(1, 2)
        branches = torch.stack([layer(base) for layer in self.branches], dim=1)
        alpha = F.softmax(self.gate(torch.cat([z, jac], -1)), -1)
        mix = (branches * alpha.transpose(1, 2).unsqueeze(2)).sum(dim=1)
        freq = torch.fft.rfft(mix.float(), dim=-1)
        bins = freq.shape[-1]
        real = F.interpolate(self.freq_real[None], size=bins, mode="linear", align_corners=True)[0]
        imag = F.interpolate(self.freq_imag[None], size=bins, mode="linear", align_corners=True)[0]
        transfer = torch.complex(real, imag)
        freq_out = torch.fft.irfft(freq * transfer[None], n=mix.size(-1), dim=-1).to(mix.dtype)
        refined = self.fuse_proj(mix + self.spec_proj(freq_out))
        beta = self.beta_logit.sigmoid() * strength
        return (base + beta * (refined - base)).transpose(1, 2)


class SurrogateForSmokeTests(nn.Module):

    def __init__(self, hidden, outputs):
        super().__init__()
        self.k = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.GELU(),
                               nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, 2 * outputs)

    def forward(self, z, query_normalized):
        B, M, C = z.shape
        u = torch.linspace(0, 1, M, device=z.device, dtype=z.dtype)
        q = query_normalized
        feats = torch.cat([z[:, None].expand(-1, q.size(1), -1, -1),
             u[None, None, :, None].expand(B, q.size(1), -1, -1),
             q[:, :, None, None].expand(-1, -1, M, 1)], dim=-1)

        integral = self.k(feats).mean(2)
        return self.head(integral)


class OriginalKRNOBridge(nn.Module):

    def __init__(self, hidden, out_channels, integral_channels=20, layers=3,
                 lifting_channels=128, kernel_hidden=32, kernel_layers=3,
                 predictive_variance=True):
        super().__init__()
        try:
            from khatriraonop import models, quadrature
        except ImportError as exc:
            raise ImportError("Install original github.com/srinathdama/ShiftingTime/krno; "
                              "surrogate is strictly for code smoke tests") from exc
        self.quadrature = quadrature
        self.predictive_variance = predictive_variance
        self.operator = models.KhatriRaoNO_v2.easy_init(
            1, in_channels=hidden, out_channels=(2 if predictive_variance else 1)*out_channels,
            lifting_channels=lifting_channels, integral_channels=integral_channels,
            n_integral_layers=layers, projection_channels=lifting_channels,
            n_hidden_units=kernel_hidden, n_hidden_layers=kernel_layers,
            nonlinearity=nn.SiLU())

    def forward(self, z, query_normalized, input_coords=None):


        B, M, C = z.shape
        if input_coords is None:
            input_coords = torch.linspace(0, 1, M, device=z.device, dtype=z.dtype)[None].expand(B, -1)
        if input_coords.shape != (B, M):
            raise ValueError("Input grid must be [B,T]")
        if query_normalized.ndim != 2 or query_normalized.size(0) != B:
            raise ValueError("Output query grid must be [B,H]")
        if not torch.all(input_coords[:, 1:] > input_coords[:, :-1]):
            raise ValueError("Input coordinates must increase strictly")
        if not torch.all(query_normalized[:, 1:] > query_normalized[:, :-1]):
            raise ValueError("Output coordinates must increase strictly")

        def invoke(sub_z, x, q):
            wi, xi = original_uneven_trapezoid(x)
            if q.numel() == 1:

                wo, xo = torch.ones_like(q), q
            else:
                wo, xo = original_uneven_trapezoid(q)
            in_grid = ([wi.to(z.device)], [xi.to(z.device)])
            out_grid = ([wo.to(z.device)], [xo.to(z.device)])


            return self.operator.super_resolution(out_grid, in_grid, sub_z)
        shared_in = torch.allclose(input_coords, input_coords[:1].expand_as(input_coords), atol=1e-7)
        shared_out = torch.allclose(query_normalized, query_normalized[:1].expand_as(query_normalized), atol=1e-7)
        if shared_in and shared_out:
            return invoke(z, input_coords[0], query_normalized[0])
        return torch.cat([invoke(z[i:i+1], input_coords[i], query_normalized[i])
                          for i in range(B)], dim=0)


class SHIFT(nn.Module):
    def __init__(self, channels, hidden=32, outputs=None, backbone="original",
                 width=32, kernel_hidden=64, integral_layers=3,
                 rev_in=False, recurrent_noise=None):
        super().__init__()
        self.channels = channels
        self.has_predictive_variance = False
        self.out_channels = outputs or channels
        self.rev_in = rev_in
        self.warp = ConditionalWarp(channels, hidden=hidden)
        self.multiscale = MultiscaleSpectral(hidden)
        if backbone == "original":
            self.backbone = OriginalKRNOBridge(hidden, hidden, width,
                                              integral_layers, 128, kernel_hidden,predictive_variance=False)
        elif backbone == "surrogate":
            self.backbone = SurrogateForSmokeTests(hidden, self.out_channels)
        else:
            raise ValueError("Unknown backbone; use original for real benchmarks")
        self.backbone_name = backbone
        if backbone=="original":
            self.mean_head=nn.Linear(hidden,self.out_channels)
            self.variance_head=nn.Linear(hidden,self.out_channels)

    def forward(self, x, mask, times, future_times, K=4, mode="conditional",
                strength=None, external_noise=None):
        strength=getattr(self,"inference_strength",1.) if strength is None else strength
        validate_input(x, mask, times, future_times)
        if self.rev_in:

            count = mask.sum(1, keepdim=True).clamp_min(1)
            avg = (x * mask).sum(1, keepdim=True) / count
            std = (((x-avg).square() * mask).sum(1, keepdim=True)/count + 1e-5).sqrt()
            x_model = torch.where(mask.bool(), (x-avg)/std, torch.zeros_like(x))
        else:
            x_model = x
        z, phi, rates, mu, lv, ds = self.warp(x_model, mask, times, K, mode, external_noise)


        span = (times[:, -1:] - times[:, :1]).clamp_min(1e-8)
        q = (future_times - times[:, :1]) / span
        component_mean, component_var = [], []
        for k in range(K):
            latent, jac = latent_resample(z, phi[k], times)
            h = self.multiscale(latent, jac, strength)
            raw = self.backbone(h, q)
            if self.backbone_name=="original":
                m,logv=self.mean_head(raw),self.variance_head(raw)
            else:
                m,logv=raw.split(self.out_channels,dim=-1)
            v = torch.exp(logv.clamp(-12, 8)).clamp_min(1e-6)
            if self.rev_in:
                if self.out_channels != self.channels:
                    raise ValueError("RevIN inverse requires outputs == channels")
                m = m * std + avg
                v = v * std.square()
            component_mean.append(m)
            component_var.append(v)
        mus = torch.stack(component_mean, 0)
        vars_ = torch.stack(component_var, 0)
        mean = mus.mean(0)
        va = vars_.mean(0)
        vt = (mus - mean).square().mean(0)
        return ShiftOutput(mean, va + vt, va, vt, mus, vars_, phi, rates, mu, lv,
                           rates*ds.unsqueeze(0), ds)

class OriginalSpatialKRNOBridge(nn.Module):


    def __init__(self, hidden=3, outputs=3, width=20, predictive_variance=False, lifting=128, layers=3, kernel_hidden=32,kernel_layers=3):
        super().__init__()
        try:
            from khatriraonop import models, quadrature
        except ImportError as exc:
            raise ImportError("Install upstream github.com/srinathdama/ShiftingTime/krno") from exc
        self.quadrature = quadrature
        self.predictive_variance = predictive_variance
        self.operator = models.KhatriRaoNO.easy_init(
            3, in_channels=hidden, out_channels=outputs * (2 if predictive_variance else 1),
            lifting_channels=lifting, integral_channels=width,
            n_integral_layers=layers, projection_channels=lifting,
            n_hidden_units=kernel_hidden, n_hidden_layers=kernel_layers, nonlinearity=nn.SiLU())

    def forward(self, z, query):
        B, T, X, Y, C = z.shape
        if (X, Y) != (32, 32):
            raise ValueError("KRNO shallow-water spatial grid is 32x32")
        if not torch.allclose(query, query[:1].expand_as(query), atol=1e-7):
            raise ValueError("Batch-specific query grid: process separately")


        fn = [self.quadrature.midpoint_vecs,
              self.quadrature.trapezoidal_vecs, self.quadrature.trapezoidal_vecs]
        if query.shape[1] != T:
            raise ValueError("5-step source and 5-step target required for shallow rollout")
        g = self.quadrature.get_quad_grid(fn,[T,X,Y],[-1.0]*3,[1.0]*3)
        g = ([w.to(z) for w in g[0]], [x.to(z) for x in g[1]])


        cart_grid = self.quadrature.quad_to_cartesian_grid(g)
        return self.operator(cart_grid, z)


class SpatialSHIFT(nn.Module):


    def __init__(self, hidden=32, width=48, surrogate=False):
        super().__init__()
        self.has_predictive_variance = False
        self.warp = ConditionalWarp(3, hidden,warp_hidden=32,rate_floor=1e-4)
        self.pixel_encoder = TemporalEncoder(3, hidden)
        self.multiscale = MultiscaleSpectral(hidden)
        if surrogate:
            raise ValueError("Spatial surrogate intentionally unavailable: install original KRNO")
        self.backbone = OriginalSpatialKRNOBridge(hidden,hidden,width=width,predictive_variance=False,lifting=256,layers=5,kernel_hidden=96,kernel_layers=5)
        self.mean_head=nn.Linear(hidden,3)
        self.variance_head=nn.Linear(hidden,3)

    def forward(self, x, mask, times, future_times, K=8, mode="conditional",strength=None):
        strength=getattr(self,"inference_strength",1.) if strength is None else strength
        B,T,H,W,C = x.shape
        assert C == 3 and (H,W) == (32,32)
        if mask.shape != x.shape: raise ValueError("Spatial masks must match field")
        xm = x.mean((2,3))
        mm = mask.amin((2,3))
        _, phi, rates, mu, lv, ds = self.warp(xm, mm, times, K, mode)

        xx=x.permute(0,2,3,1,4).reshape(B*H*W,T,C)
        mk=mask.permute(0,2,3,1,4).reshape(B*H*W,T,C)
        tt=times[:,None,None,:].expand(B,H,W,T).reshape(B*H*W,T)
        pix=self.pixel_encoder(xx,mk,tt)
        pix=pix.reshape(B,H,W,T,-1).permute(0,3,1,2,4)
        q=(future_times-times[:,:1])/(times[:,-1:]-times[:,:1]).clamp_min(1e-8)
        mean_k,var_k=[],[]
        for k in range(K):
            z=pix.permute(0,2,3,1,4).reshape(B*H*W,T,-1)
            p=phi[k][:,None,None,:].expand(B,H,W,T).reshape(B*H*W,T)
            latent,jac=latent_resample(z,p,tt)
            fusion=self.multiscale(latent,jac,strength)
            fusion=fusion.reshape(B,H,W,T,-1).permute(0,3,1,2,4)
            raw=self.backbone(fusion,q)
            m,lvraw=self.mean_head(raw),self.variance_head(raw)
            mean_k.append(m)
            var_k.append(lvraw.clamp(-12,8).exp().clamp_min(1e-6))
        mus=torch.stack(mean_k)
        vars_=torch.stack(var_k)
        mean=mus.mean(0)
        va=vars_.mean(0)
        vt=(mus-mean).square().mean(0)
        return ShiftOutput(mean, va+vt,va,vt,mus,vars_,phi,rates,mu,lv,rates*ds[None],ds)


def _as_shift_output(mus, vars_, phi, rates, mu, logvar, ds):
    mean = mus.mean(0)
    v_value = vars_.mean(0)
    v_time = (mus - mean).square().mean(0)
    return ShiftOutput(mean, v_value + v_time, v_value, v_time, mus, vars_,
                       phi, rates, mu, logvar, rates * ds[None], ds)


class KRNOReference(nn.Module):


    def __init__(self, channels, width=20, kernel_hidden=32,
                 integral_layers=3, probabilistic=False, mask_features=False):
        super().__init__()
        self.channels=channels
        self.mask_features=mask_features
        self.probabilistic=probabilistic
        self.has_predictive_variance=probabilistic
        self.backbone=OriginalKRNOBridge(
            channels*(2 if mask_features else 1), channels,
            integral_channels=width, layers=integral_layers,
            kernel_hidden=kernel_hidden, predictive_variance=probabilistic)

    def forward(self, x, mask, times, future_times, K=1, mode='fixed', **kwargs):
        validate_input(x,mask,times,future_times)
        ds=times[:,1:]-times[:,:-1]
        span=times[:,-1:]-times[:,:1]
        coords=(times-times[:,:1])/span
        q=(future_times-times[:,:1])/span
        features=(torch.cat([x*mask,mask],dim=-1)
                  if self.mask_features else x*mask)
        raw=self.backbone(features, q, input_coords=coords)
        if self.probabilistic:
            mean,lv=raw.split(self.channels,-1)
            var=lv.clamp(-12,8).exp().clamp_min(1e-6)
        else:
            mean=raw
            var=torch.ones_like(mean)
        mus=mean.unsqueeze(0).expand(K,*mean.shape)
        vars_=var.unsqueeze(0).expand(K,*var.shape)
        rate=torch.ones((K,*ds.shape),device=x.device,dtype=x.dtype)
        z=torch.zeros_like(ds)
        return _as_shift_output(mus,vars_,coords[None].expand(K,-1,-1),rate,z,z,ds)


class KRNOWarpOnly(nn.Module):


    def __init__(self,channels,width=20,kernel_hidden=32,integral_layers=3,
                 warp_hidden=64,probabilistic=False,mask_features=False):
        super().__init__()
        self.channels=channels
        self.mask_features=mask_features
        self.probabilistic=probabilistic
        self.has_predictive_variance=probabilistic
        self.warp=ConditionalWarp(channels,hidden=32,warp_hidden=warp_hidden)
        self.backbone=OriginalKRNOBridge(
            channels*(2 if mask_features else 1),channels,
            integral_channels=width,layers=integral_layers,
            kernel_hidden=kernel_hidden,predictive_variance=probabilistic)

    def forward(self,x,mask,times,future_times,K=4,mode='conditional',external_noise=None,**kw):
        validate_input(x,mask,times,future_times)
        _,phi,rates,mu,lv,ds=self.warp(x,mask,times,K,mode,external_noise)
        span=times[:,-1:]-times[:,:1]
        original_coords=(times-times[:,:1])/span
        queries=(future_times-times[:,:1])/span
        features=(torch.cat([x*mask,mask],dim=-1)
                  if self.mask_features else x*mask)
        means=[];variances=[]
        for k in range(K):
            if mode=='fixed':
                warped_input,coords=features,original_coords
            else:


                warped_input,_=latent_resample(features,phi[k],times)
                coords=torch.linspace(0,1,warped_input.size(1),
                        device=x.device,dtype=x.dtype)[None].expand(x.size(0),-1)
            raw=self.backbone(warped_input,queries,input_coords=coords)
            if self.probabilistic:
                m,logv=raw.split(self.channels,-1)
                v=logv.clamp(-12,8).exp().clamp_min(1e-6)
            else:
                m=raw
                v=torch.ones_like(m)
            means.append(m);variances.append(v)
        return _as_shift_output(torch.stack(means),torch.stack(variances),
                                phi,rates,mu,lv,ds)


class SpatialKRNOReference(nn.Module):

    def __init__(self,width=20,probabilistic=False):
        super().__init__()
        self.probabilistic=probabilistic
        self.has_predictive_variance=probabilistic
        self.backbone=OriginalSpatialKRNOBridge(3,3,width,probabilistic)

    def forward(self,x,mask,times,future_times,K=1,mode='fixed',**kw):
        if x.ndim!=5 or x.size(-1)!=3 or x.shape[2:4]!=(32,32):
            raise ValueError('Expected [B,5,32,32,3]')
        if x.size(1)!=5 or future_times.size(1)!=5:
            raise ValueError('Shallow original KRNO uses 5->5 windows')
        raw=self.backbone(x*mask,future_times)
        if self.probabilistic:
            mean,lv=raw.split(3,-1)
            var=lv.clamp(-12,8).exp().clamp_min(1e-6)
        else:mean=raw;var=torch.ones_like(raw)
        ds=times[:,1:]-times[:,:-1]
        phi=(times-times[:,:1])/(times[:,-1:]-times[:,:1])
        rates=torch.ones((K,*ds.shape),device=x.device,dtype=x.dtype)
        return _as_shift_output(mean[None].expand(K,*mean.shape),
                                var[None].expand(K,*var.shape),
                                phi[None].expand(K,-1,-1),rates,
                                torch.zeros_like(ds),torch.zeros_like(ds),ds)


class SpatialKRNOWarpOnly(nn.Module):

    def __init__(self,width=20,probabilistic=False):
        super().__init__()
        self.warp=ConditionalWarp(3,hidden=32)
        self.backbone=OriginalSpatialKRNOBridge(3,3,width,probabilistic)
        self.probabilistic=probabilistic
        self.has_predictive_variance=probabilistic

    def forward(self,x,mask,times,future_times,K=4,mode='conditional',**kw):
        if x.ndim!=5 or x.shape[2:]!=(32,32,3) or x.size(1)!=5:
            raise ValueError('Shallow Water requires [B,5,32,32,3]')
        if future_times.size(1)!=5:raise ValueError('Shallow forecast block must be 5')
        if not torch.all(times[:,1:]>times[:,:-1]):raise ValueError('Time order')
        vals=(x*mask).mean(dim=(2,3))
        summary_mask=mask.amin(dim=(2,3))
        _,phi,rates,mu,lv,ds=self.warp(vals,summary_mask,times,K,mode)
        B,T,H,W,C=x.shape
        raw_hist=x*mask
        means=[];variances=[]
        for k in range(K):
            if mode=='fixed':z=raw_hist
            else:
                pix=raw_hist.permute(0,2,3,1,4).reshape(B*H*W,T,C)
                pp=phi[k][:,None,None,:].expand(B,H,W,T).reshape(B*H*W,T)
                tt=times[:,None,None,:].expand(B,H,W,T).reshape(B*H*W,T)
                z,_=latent_resample(pix,pp,tt)
                z=z.reshape(B,H,W,T,C).permute(0,3,1,2,4)
            raw=self.backbone(z,future_times)
            if self.probabilistic:
                m,l=raw.split(3,-1)
                v=l.clamp(-12,8).exp().clamp_min(1e-6)
            else:m=raw;v=torch.ones_like(raw)
            means.append(m);variances.append(v)
        return _as_shift_output(torch.stack(means),torch.stack(variances),
                                phi,rates,mu,lv,ds)

class KRNOControl(nn.Module):


    def __init__(self, channels, hidden=32, width=32, kernel_hidden=64, rev_in=False):
        super().__init__()
        self.channels=channels
        self.has_predictive_variance = False
        self.encoder=TemporalEncoder(channels,hidden)
        self.backbone=OriginalKRNOBridge(hidden,channels,width,3,128,kernel_hidden)
        self.rev_in=rev_in

    def forward(self,x,mask,times,future_times,K=4,mode='fixed',**kw):
        validate_input(x,mask,times,future_times)
        if self.rev_in:
            count=mask.sum(1,keepdim=True).clamp_min(1)
            avg=(x*mask).sum(1,keepdim=True)/count
            std=(((x-avg).square()*mask).sum(1,keepdim=True)/count+1e-5).sqrt()
            inp=(x-avg)/std*mask
        else:inp=x
        encoded=self.encoder(inp,mask,times)
        span=(times[:,-1:]-times[:,:1]).clamp_min(1e-8)
        q=(future_times-times[:,:1])/span
        raw=self.backbone(encoded,q)
        mu,lv=raw.split(self.channels,-1)
        var=lv.clamp(-12,8).exp().clamp_min(1e-6)
        if self.rev_in:
            mu=mu*std+avg;var=var*std.square()
        mus=mu[None].expand(K,-1,-1,-1)
        vars_=var[None].expand(K,-1,-1,-1)
        phi=(times-times[:,:1])/span
        r=torch.ones((K,times.shape[0],times.shape[1]-1),device=x.device)
        zeros=torch.zeros_like(r[0])
        return ShiftOutput(mu,var,var,torch.zeros_like(var),mus,vars_,
                           phi[None].expand(K,-1,-1),r,zeros,zeros,
                           r*(times[:,1:]-times[:,:-1])[None],times[:,1:]-times[:,:-1])

class SpatialKRNOControl(nn.Module):

    def __init__(self,width=48):
        super().__init__()
        self.has_predictive_variance = False
        self.backbone=OriginalSpatialKRNOBridge(hidden=3,outputs=3,width=width,predictive_variance=True)

    def forward(self,x,mask,times,future_times,K=8,mode='fixed'):
        if x.ndim!=5 or x.size(-1)!=3 or x.shape[2:4]!=(32,32):
            raise ValueError('Spatial KRNO requires [B,T,32,32,3]')
        q=(future_times-times[:,:1])/(times[:,-1:]-times[:,:1]).clamp_min(1e-8)
        raw=self.backbone(x*mask,q)
        mean,logvar=raw.split(3,-1)
        var=logvar.clamp(-12,8).exp().clamp_min(1e-6)
        mus=mean[None].expand(K,-1,-1,-1,-1,-1)
        variances=var[None].expand_as(mus)
        phi=(times-times[:,:1])/(times[:,-1:]-times[:,:1]).clamp_min(1e-8)
        ds=times[:,1:]-times[:,:-1]
        rate=torch.ones(K,*ds.shape,device=x.device,dtype=x.dtype)
        zeros=torch.zeros_like(ds)
        return ShiftOutput(mean,var,var,torch.zeros_like(var),mus,variances,
                           phi[None].expand(K,-1,-1),rate,zeros,zeros,rate*ds[None],ds)

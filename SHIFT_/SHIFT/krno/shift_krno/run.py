import argparse
import copy
import hashlib
import json
import random
import sys
import subprocess
import importlib.util
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from .data import (DARTS,M4_HORIZONS,load_darts,load_m4,load_crypto,
                   load_trajectory,load_irregular,load_shallow)
from .model import (SHIFT, SpatialSHIFT, KRNOControl, SpatialKRNOControl,
                    KRNOReference, KRNOWarpOnly, SpatialKRNOReference, SpatialKRNOWarpOnly)
from .losses import forecast_loss,masked_mean,nmae,variable_error_sums,macro_from_sums,manuscript_objective,quadrature_mse,shallow_field_losses
from .evaluate import (tensors,eval_darts,eval_m4,eval_crypto,eval_trajectory,
                       eval_irregular,eval_shallow)


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def load_dataset(a,refit=False):
    name=a.dataset
    if name.startswith('darts/'):
        return load_darts(a.data_root,name.split('/',1)[1],a.lookback or 48,a.block or 12,refit)
    if name.startswith('m4/'):
        return load_m4(a.data_root,name.split('/',1)[1],a.lookback,a.block,refit,a.val_fraction,a.train_stride)
    if name=='crypto':return load_crypto(a.data_root,a.lookback or 80,a.block or 15,
                                         a.train_stride or 100,a.test_stride or 15,refit)
    if name=='trajectory':return load_trajectory(a.data_root,a.lookback or 30,a.block or 15,refit,a.train_stride or 2)
    if name in ('mimic','ushcn','activity','mujoco','synthetic'):
        return load_irregular(a.data_root,name,a.seed,a.missing_rate)
    if name=='shallow':return load_shallow(a.data_root)
    raise ValueError(f"Unknown dataset {name}")


def create_model(a, channels):

    if a.dataset == 'shallow':
        if a.architecture == 'krno_reference':
            return SpatialKRNOReference(width=a.width, probabilistic=a.probabilistic)
        if a.architecture == 'krno_warp':
            return SpatialKRNOWarpOnly(width=a.width, probabilistic=a.probabilistic)
        if a.architecture == 'krno_control':
            return SpatialKRNOControl(width=a.width)
        return SpatialSHIFT(hidden=a.hidden, width=a.width)
    if a.architecture == 'krno_reference':
        return KRNOReference(channels,width=a.width,kernel_hidden=a.kernel_hidden,
                             integral_layers=a.integral_layers,probabilistic=a.probabilistic,
                             mask_features=a.dataset in ('mimic','ushcn','activity','mujoco'))
    if a.architecture == 'krno_warp':
        return KRNOWarpOnly(channels,width=a.width,kernel_hidden=a.kernel_hidden,
                            integral_layers=a.integral_layers,probabilistic=a.probabilistic,
                            mask_features=a.dataset in ('mimic','ushcn','activity','mujoco'))
    if a.architecture == 'krno_control':
        return KRNOControl(channels,hidden=a.hidden,width=a.width,
                           kernel_hidden=a.kernel_hidden,rev_in=a.revin)
    return SHIFT(channels=channels,hidden=a.hidden,backbone=a.backbone,
                 width=a.width,kernel_hidden=a.kernel_hidden,
                 integral_layers=a.integral_layers,rev_in=a.revin)


def profile(model):
    return dict(total_params=sum(x.numel() for x in model.parameters()),
                trainable_params=sum(x.numel() for x in model.parameters() if x.requires_grad))


def objective_params(a):
    return dict(smooth=a.smooth,rate=a.rate,kl=a.kl,length=a.length)


def step_loss(a,out,y,ym,batch=None):
    regularize=a.architecture in ('krno_warp','shift') and a.mode!='fixed' and a.reg_strength>0
    kw={k:v*a.reg_strength for k,v in objective_params(a).items()} if regularize else None
    if kw is not None:kw['shape']=a.shape_penalty
    if a.objective=='manuscript':
        pp=tt=None
        if a.dataset.startswith('m4/'):
            stats=a._metadata['stats']
            idx=batch['series'].tolist()
            mu=y.new_tensor([stats[i]['mean'] for i in idx])[:,None,None]
            sd=y.new_tensor([stats[i]['std'] for i in idx])[:,None,None]
            pp=out.mean*sd+mu;tt=y*sd+mu
        return manuscript_objective(out,y,ym,a.dataset,a._metadata.get('lookback'),pp,tt,
            nll_warmup=min(1.,(a._epoch+1)/a.nll_warmup_epochs) if a.nll_warmup_epochs else None,
            tail_steps=a.tail_steps,variance_floor=a.variance_floor,
            diversity_margin=(a.diversity*a.reg_strength,a.diversity_margin) if regularize and a.diversity else None,reg_kwargs=kw)
    return forecast_loss(out,y,ym,objective=a.objective,rate=a.nll_weight,reg_kwargs=kw,macro=a.dataset in ('mimic','ushcn','activity'))


def per_batch_forecast(model,batch,a,K,device):
    x,m,t,q,y,ym=tensors(batch,device)
    if a.input_noise>0 and model.training:


        x=x+torch.randn_like(x)*a.input_noise*m
    if a.dropout>0:

        x=torch.nn.functional.dropout(x,p=a.dropout,training=model.training)*m
    out=model(x,m,t,q,K=K,mode=a.mode,strength=a.strength) if getattr(a,"architecture",None)=="shift" else model(x,m,t,q,K=K,mode=a.mode)
    return out,y,ym


def validate(model,loader,a,device):
    model.eval();num=0.;den=0.;sums=0.;counts=0.;batch_losses=[]
    macro=a.dataset in ('mimic','ushcn','activity')
    with torch.no_grad():
        for batch in loader:
            out,y,ym=per_batch_forecast(model,batch,a,a.eval_samples,device)
            pred=out.mean
            if macro:
                se,_,n=variable_error_sums(pred,y,ym);sums=sums+se.double().cpu();counts=counts+n.double().cpu()
            elif a.dataset=='shallow':
                batch_losses.append(float(shallow_field_losses(pred,y).mean()))
            elif a.dataset.startswith('m4/') or a.dataset in ('crypto','trajectory'):
                batch_losses.append(float(quadrature_mse(pred,y,a._metadata['lookback'])))
            else:
                num+=float(((pred-y).square()*ym).sum());den+=float(ym.sum())
    if macro:return float(macro_from_sums(sums,counts))
    if batch_losses:return float(np.mean(batch_losses))**.5
    if den==0:raise ValueError('No observed validation targets')
    return num/den


def train(model,bundle,a,device):
    if a.refit and not a.refit_epochs:
        raise ValueError('--refit needs preselected --refit-epochs; no train/test epoch tuning')
    if bundle.metadata.get('variable_length') and a.batch_size!=1:raise ValueError('Variable-length original examples require batch size 1')
    tr=DataLoader(bundle.train,batch_size=a.batch_size,shuffle=True,
                  generator=torch.Generator().manual_seed(a.seed),num_workers=0)
    va=DataLoader(bundle.val,batch_size=a.batch_size,shuffle=False,num_workers=0)
    if len(tr)==0:raise ValueError('No training windows; reduce lookback/block')
    if len(va)==0 and not a.refit:raise ValueError('No validation windows; cannot select checkpoint')
    opt={"adamw":torch.optim.AdamW,"adam":torch.optim.Adam}[a.optimizer](model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    sched=(torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.eta_min) if a.scheduler=="cosine" else torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=a.scheduler_factor,patience=a.scheduler_patience))
    best=float('inf');best_state=None;history=[];bad=0
    for epoch in range(a.refit_epochs if a.refit else a.epochs):
        a._epoch=epoch
        model.train();total=0.;n=0;opt.zero_grad()
        for step,b in enumerate(tr):
            out,y,ym=per_batch_forecast(model,b,a,a.train_samples,device)
            group_start=(step//a.grad_accum)*a.grad_accum
            group_size=min(a.grad_accum,len(tr)-group_start)
            loss=step_loss(a,out,y,ym,b)/group_size
            if not torch.isfinite(loss):raise FloatingPointError('Training loss non-finite')
            loss.backward()
            total+=float(loss.detach())*group_size;n+=1
            if ((step+1)%a.grad_accum==0) or (step+1==len(tr)):
                clip_grad_norm_(model.parameters(),a.clip)
                opt.step();opt.zero_grad()
        metric=validate(model,va,a,device) if not a.refit else total/max(n,1)
        if a.scheduler=="cosine":sched.step()
        else:sched.step(metric)
        history.append(dict(epoch=epoch+1,train_objective=total/max(n,1),
                            validation_metric=metric))
        if metric<best-1e-10:
            best=metric;bad=0
            best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:bad+=1
        if bad>=a.patience and not a.refit:break
    if a.refit:

        best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return dict(best_validation_metric=best,history=history,epochs_trained=len(history))


def upstream_identity(root):

    if root is None:return dict(verified=False,reason='No --upstream-root supplied')
    root=Path(root).resolve()
    if not root.exists():raise FileNotFoundError(root)
    try:
        sha=subprocess.run(['git','rev-parse','HEAD'],cwd=root,
                 check=True,capture_output=True,text=True).stdout.strip()
        dirty=subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=root,
                 check=True,capture_output=True,text=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError) as exc:
        raise RuntimeError('Official KRNO source must be an inspectable git clone') from exc
    if dirty:raise ValueError('Upstream source has uncommitted modifications; pin a clean commit')
    package=importlib.util.find_spec('khatriraonop')
    if package is None:raise ImportError('Original khatriraonop is not installed')
    actual=Path(package.origin).resolve()
    if root not in actual.parents:
        raise RuntimeError('Imported khatriraonop is NOT from --upstream-root; '
                           'pip install -e <upstream>/krno and verify module path')
    return dict(verified=True,repository=str(root),commit=sha,
                imported_package_origin=str(package.origin),source_clean=True)


def file_inventory(root,max_files=300):
    root=Path(root)
    paths=[p for p in root.rglob('*') if p.is_file()]
    if len(paths)>max_files:
        raise ValueError(f'{len(paths)} source files exceed auditable file cap {max_files}; '
                         'prepare a minimal immutable benchmark bundle or increase the cap explicitly')
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    return {str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in paths}


def evaluate(model,bundle,a):
    if a.backbone!='original' or a.test_forbidden:
        raise PermissionError('Scientific test evaluation forbidden with surrogate or --test-forbidden')
    if a.dataset.startswith('darts/'):return eval_darts(model,bundle,a.eval_samples,a.mode,probabilistic_audit=a.probabilistic_audit)
    if a.dataset.startswith('m4/'):return eval_m4(model,bundle,a.eval_samples,a.mode,probabilistic_audit=a.probabilistic_audit)
    if a.dataset=='crypto':return eval_crypto(model,bundle,a.eval_samples,a.mode,probabilistic_audit=a.probabilistic_audit)
    if a.dataset=='trajectory':return eval_trajectory(model,bundle,a.eval_samples,a.mode,probabilistic_audit=a.probabilistic_audit)
    if a.dataset=='shallow':return eval_shallow(model,bundle,a.eval_samples,a.mode,probabilistic_audit=a.probabilistic_audit)
    return eval_irregular(model,bundle,a.eval_samples,a.mode,
                          probabilistic_audit=a.probabilistic_audit)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,help='darts/AirPassengers, m4/Weekly, crypto, trajectory, mimic, ushcn, activity, mujoco, synthetic, shallow')
    p.add_argument('--data-root',required=True)
    p.add_argument('--missing-rate',type=float,choices=[0.,.3,.5,.7])
    p.add_argument('--out',default='runs/shift')
    p.add_argument('--seed',type=int,default=2021)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--grad-accum',type=int,default=1)
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--clip',type=float,default=1.0)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--weight-decay',type=float,default=1e-6)
    p.add_argument('--lookback',type=int)
    p.add_argument('--block',type=int)
    p.add_argument('--train-stride',type=int)
    p.add_argument('--test-stride',type=int)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--width',type=int,default=20)
    p.add_argument('--kernel-hidden',type=int,default=32)
    p.add_argument('--integral-layers',type=int,default=3)
    p.add_argument('--train-samples',type=int,default=4)
    p.add_argument('--eval-samples',type=int,default=16)
    p.add_argument('--objective',choices=['mse','mix_nll','component_nll','mse_plus_nll','manuscript'],default='mse')
    p.add_argument('--val-fraction',type=float)
    p.add_argument('--optimizer',choices=['adam','adamw'],default='adamw')
    p.add_argument('--scheduler',choices=['plateau','cosine'],default='plateau')
    p.add_argument('--scheduler-factor',type=float,default=.5)
    p.add_argument('--scheduler-patience',type=int,default=2)
    p.add_argument('--eta-min',type=float,default=0.)
    p.add_argument('--shape-penalty',action='store_true')
    p.add_argument('--diversity',type=float,default=0.)
    p.add_argument('--diversity-margin',type=float)
    p.add_argument('--variance-floor',type=float)
    p.add_argument('--tail-steps',type=int)
    p.add_argument('--nll-warmup-epochs',type=int)
    p.add_argument('--strength',type=float,default=1.)
    p.add_argument('--allow-reconstructed-objective',action='store_true')
    p.add_argument('--nll-weight',type=float,default=.01)
    p.add_argument('--reg-strength',type=float,default=0.)
    p.add_argument('--smooth',type=float,default=.002)
    p.add_argument('--rate',type=float,default=.00001)
    p.add_argument('--length',type=float,default=.005)
    p.add_argument('--kl',type=float,default=0.)
    p.add_argument('--architecture',choices=['krno_reference','krno_warp','shift','krno_control'],default='shift')
    p.add_argument('--probabilistic',action='store_true',help='Identical 2*C Gaussian heads in paired arms')
    p.add_argument('--probabilistic-audit',action='store_true',help='Evaluate masked finite-mixture metrics, no normal shortcut')
    p.add_argument('--mode',choices=['conditional','deterministic','fixed','random','unconditional','shuffled'],default='conditional')
    p.add_argument('--donor-audit',action='store_true')
    p.add_argument('--input-noise',type=float,default=0.)
    p.add_argument('--dropout',type=float,default=0.)
    p.add_argument('--revin',action='store_true')
    p.add_argument('--backbone',choices=['original','surrogate'],default='original')
    p.add_argument('--upstream-root',help='Clean git checkout of original ShiftingTime repository; required for --test')
    p.add_argument('--refit',action='store_true',help='Refit using validation-chosen fixed epochs')
    p.add_argument('--refit-epochs',type=int,help='Mandatory preselected epochs for refit')
    p.add_argument('--test',action='store_true',help='Evaluate selected checkpoint on test')
    p.add_argument('--checkpoint',help='Existing best.pt for test-only, no retraining')
    p.add_argument('--validate-only',action='store_true',help='Check data contract without model')
    p.add_argument('--test-forbidden',action='store_true',help='Disable all test evaluation (tuning)')
    p.add_argument('--unsafe-smoke',action='store_true',help='Allows surrogate smoke/training but never final test')
    return p


def main(argv=None):
    a=parser().parse_args(argv)


    reported={'mimic','ushcn','activity','mujoco','crypto','trajectory','shallow',
              'm4/Quarterly','m4/Weekly','m4/Daily'}
    if a.dataset not in reported and not (a.dataset=='synthetic' and a.test_forbidden):
        raise ValueError('Only SHIFT-reported datasets are available as benchmarks in this package')
    if a.backbone=='surrogate' and not a.unsafe_smoke:
        raise PermissionError('surrogate forbidden except explicit --unsafe-smoke; cannot report benchmark numbers')
    if a.test and not a.upstream_root:
        raise ValueError('For scientific test evaluation supply pinned clean --upstream-root')
    if a.test and (a.test_forbidden or a.backbone!='original'):
        raise PermissionError('Test is locked for tuning/surrogate runs')
    if a.architecture in ('krno_reference','krno_warp') and a.revin:
        raise ValueError('Raw operator comparison cannot use unshared RevIN')
    if a.architecture in ('krno_reference','krno_warp') and \
       a.objective in ('mix_nll','component_nll','mse_plus_nll') and not a.probabilistic:
        raise ValueError('Probabilistic loss requires --probabilistic learned head')
    if a.probabilistic_audit and not a.probabilistic and a.architecture in ('krno_reference','krno_warp'):
        raise ValueError('Enable --probabilistic to perform probabilistic audit')
    if a.dataset=='shallow' and a.block not in (None,5):
        raise ValueError('Official Shallow Water protocol uses five-step blocks')
    if min(a.train_samples,a.eval_samples,a.grad_accum,a.epochs,a.batch_size)<1:raise ValueError("Positive sample counts and budgets required")
    if a.test and a.dataset=="crypto" and a.block not in (None,15):raise ValueError("Selected Crypto test uses a 15-step block; tune shorter blocks without test")
    if a.objective=="mix_nll" and a.architecture in ("shift","krno_warp") and a.mode in ("conditional","random","unconditional") and (a.train_samples<2 or a.eval_samples<2):raise ValueError("Stochastic mixture study requires at least two samples")
    if a.donor_audit and (not a.test or not a.checkpoint):raise ValueError("Donor audit requires --test and --checkpoint")
    set_seed(a.seed)
    bundle=load_dataset(a,refit=a.refit)
    a._metadata=bundle.metadata
    if a.objective=="manuscript":
        if a.architecture!="shift":raise ValueError("Manuscript objective is for Full SHIFT only")
        if a.diversity and a.diversity_margin is None:raise ValueError("Supply the original diversity formulation/margin")
        if a.dataset=="ushcn" and not a.nll_warmup_epochs:raise ValueError("Supply original NLL warm-up")
        if a.dataset=="crypto" and a.tail_steps is None:raise ValueError("Supply original tail length")
        if a.dataset=="mujoco" and a.variance_floor is None:raise ValueError("Supply original uncertainty floor")
        if not a.allow_reconstructed_objective:raise ValueError("The manuscript omits exact implementation details. This is a reconstruction, not verified reproduction; explicitly opt in with --allow-reconstructed-objective")
    if not 0<=a.strength<=1:raise ValueError("strength must be in [0,1]")
    if a.validate_only:
        print(json.dumps(bundle.metadata,indent=2,default=str))
        print('Train:',len(bundle.train),'Validation:',len(bundle.val))
        return
    output_resolved=Path(a.out).resolve()
    data_root_resolved=Path(a.data_root).resolve()
    if output_resolved==data_root_resolved or data_root_resolved in output_resolved.parents:
        raise ValueError('Experiment output cannot live within data-root; data hash would change')
    source_identity=upstream_identity(a.upstream_root)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=create_model(a,bundle.channels).to(device)
    if a.architecture in ("shift","krno_control"):
        if a.architecture=="shift":model.inference_strength=a.strength
        model.has_predictive_variance=(a.objective in ("mix_nll","component_nll") or (a.objective=="mse_plus_nll" and a.nll_weight>0) or (a.objective=="manuscript" and a.dataset in ("ushcn","mujoco","crypto")))
    if a.probabilistic_audit and not getattr(model,"has_predictive_variance",False):raise ValueError("Variance head is not trained by selected objective")
    output=Path(a.out);output.mkdir(parents=True,exist_ok=True)
    if not a.checkpoint and (output/'best.pt').exists():raise FileExistsError('Run directory already contains a checkpoint; choose a fresh output')
    details=dict(utc=datetime.now(timezone.utc).isoformat(),
                 arguments={k:v for k,v in vars(a).items() if not k.startswith('_')},
                 data_contract=bundle.metadata,
                 data_inventory=file_inventory(a.data_root),
                 parameter_count=profile(model),
                 torch_version=torch.__version__,
                 upstream_KRNO=source_identity,
                 original_KRNO_package_required=a.backbone=='original',
                 scientific_comparison_allowed=False,
                 reproduction_status="reconstructed local implementation; not the native KRNO baseline or verified original SHIFT run",
                 reviewer_notice='No scores reproduced without official data and model execution')
    if a.checkpoint:
        cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        frozen=cp['metadata']
        old=frozen['arguments']
        required=('dataset','lookback','block','hidden','width','kernel_hidden',
                  'integral_layers','revin','architecture','backbone','mode','probabilistic','objective','train_stride','test_stride','val_fraction','train_samples','eval_samples','strength','shape_penalty','diversity','diversity_margin','variance_floor','tail_steps','nll_warmup_epochs','seed','missing_rate','input_noise','dropout','nll_weight','reg_strength','smooth','rate','kl','length','optimizer','scheduler','scheduler_factor','scheduler_patience','grad_accum','batch_size','lr','epochs','patience','weight_decay','clip','eta_min','refit','refit_epochs')
        for field in required:
            if old[field]!=getattr(a,field):
                raise ValueError(f'Checkpoint protocol mismatch for {field}: {old[field]} != {getattr(a,field)}')
        if frozen['data_inventory']!=details['data_inventory']:
            raise ValueError('Checkpoint data file hash differs from evaluation directory')
        model.load_state_dict(cp['weights'])
        details['checkpoint_source']=str(a.checkpoint)
        details['training']=frozen.get('training',{})
    else:
        fit=train(model,bundle,a,device)
        details['training']=fit
        torch.save({'weights':model.state_dict(),'metadata':details},output/'best.pt')
    if a.test:
        details['test_results']=evaluate(model,bundle,a)
        if a.donor_audit:
            if a.architecture not in ('shift','krno_warp') or a.mode!='conditional' or not a.checkpoint:
                raise ValueError('Donor audit requires a trained conditional-warp checkpoint')
            from .reviewer import donor_evaluate
            details['independent_donor_test']=donor_evaluate(model,bundle,a,evaluate)

    (output/'audit.json').write_text(json.dumps(details,indent=2,default=str))
    print(json.dumps(dict(output=str(output),parameter_count=details['parameter_count'],
                          validation=details['training'].get('best_validation_metric'),
                          test=details.get('test_results','not evaluated')),indent=2))


if __name__=='__main__':main()

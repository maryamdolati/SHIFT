from dataclasses import dataclass
from pathlib import Path
import json
from bisect import bisect_right
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

M4_HORIZONS={"Yearly":6,"Quarterly":8,"Monthly":18,
             "Weekly":13,"Daily":14,"Hourly":48}
DARTS={"AirPassengers":"AirPassengers.csv", "AusBeer":"ausbeer.csv",
       "GasRateCO2":"gasrate_co2.csv", "HeartRate":"heart_rate.csv",
       "MonthlyMilk":"monthly-milk.csv", "Sunspots":"monthly-sunspots.csv",
       "Wine":"wineind.csv", "Wooly":"woolyrnq.csv"}


def _check_finite(a, label):
    if not np.isfinite(a).all():
        raise ValueError(f"{label} contains invalid values; missingness requires explicit mask")


def read_processed_timeseries(path):
    a=np.load(path,allow_pickle=False)
    if a.ndim==1: a=a[:,None]
    if a.ndim!=2: raise ValueError(f"Expected [time,channels]: {path} {a.shape}")
    _check_finite(a,str(path))
    return a.astype('float32')


def _f32(a): return torch.as_tensor(np.asarray(a,dtype=np.float32))


class WindowDataset(Dataset):


    def __init__(self, series, origins, length, horizon, masks=None, timestamps=None):
        self.series=series
        self.origins=list(origins)
        self.length=int(length)
        self.horizon=int(horizon)
        self.masks=masks
        self.times=timestamps
        for i, o in self.origins:
            if o < length or o+horizon > len(series[i]):
                raise ValueError(f"Out-of-bounds window for series {i}: {o}")

    def __len__(self): return len(self.origins)

    def __getitem__(self,index):
        i,o=self.origins[index]
        x=self.series[i][o-self.length:o]
        y=self.series[i][o:o+self.horizon]
        m=(self.masks[i][o-self.length:o] if self.masks else np.ones_like(x))
        ym=(self.masks[i][o:o+self.horizon] if self.masks else np.ones_like(y))
        t=(self.times[i][o-self.length:o] if self.times else
           np.arange(o-self.length,o,dtype=np.float32))
        q=(self.times[i][o:o+self.horizon] if self.times else
           np.arange(o,o+self.horizon,dtype=np.float32))
        return {"x":_f32(x),"mask":_f32(m),"t":_f32(t),
                "q":_f32(q),"y":_f32(y),"ymask":_f32(ym),
                "series":torch.tensor(i),"origin":torch.tensor(o)}


def origins_between(i,L,P,start,end,stride=1):
    return [(i,o) for o in range(max(L,start),end-P+1,stride)]


class ImplicitWindows(Dataset):

    def __init__(self,series,segments,length,horizon):
        self.series=series
        self.segments=[];self.cumulative=[];total=0
        self.length=length;self.horizon=horizon
        for i,start,stop,stride in segments:
            start=max(start,length)
            count=max(0,(stop-horizon-start)//stride+1)
            if count:
                self.segments.append((i,start,stride,count))
                total+=count;self.cumulative.append(total)

    def __len__(self):return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self,index):
        if index<0 or index>=len(self):raise IndexError(index)
        s=bisect_right(self.cumulative,index)
        old=0 if s==0 else self.cumulative[s-1]
        i,start,stride,count=self.segments[s]
        o=start+(index-old)*stride
        x=self.series[i][o-self.length:o]
        y=self.series[i][o:o+self.horizon]
        return dict(x=_f32(x),mask=torch.ones_like(_f32(x)),
            t=torch.arange(o-self.length,o,dtype=torch.float32),
            q=torch.arange(o,o+self.horizon,dtype=torch.float32),
            y=_f32(y),ymask=torch.ones_like(_f32(y)),
            series=torch.tensor(i),origin=torch.tensor(o))


@dataclass
class PreparedData:
    train: Dataset
    val: Dataset
    test: Dataset | None
    channels: int
    metadata: dict
    series: object=None


def _pd_csv_value(path):
    frame=pd.read_csv(path)
    candidates=[c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    if len(candidates)!=1:
        raise ValueError(f"Expected 1 numeric series in {path}, got {candidates}")
    a=frame[candidates[0]].to_numpy(dtype=np.float32)
    _check_finite(a,path)
    return a


def load_darts(root,name,lookback=48,pred=12,final_refit=False):
    if name not in DARTS: raise ValueError(f"Unknown Darts series {name}")
    p=Path(root)/DARTS[name]
    arr=_pd_csv_value(p)
    if name=="HeartRate":arr=arr[::2]
    if name=="Sunspots":arr=arr[::4]
    n=len(arr); train_end=int(0.6*n); val_end=int(0.8*n)
    if train_end<lookback+pred or val_end-train_end<pred:
        raise ValueError(f"Darts {name} is too short for L={lookback} P={pred}")

    norm_end=val_end if final_refit else train_end
    mu=arr[:norm_end].mean();std=max(arr[:norm_end].std(),1e-6)
    z=((arr-mu)/std).astype(np.float32)[:,None]
    trains=[(0,o) for o in range(lookback,(val_end if final_refit else train_end)-pred+1)]
    vals=origins_between(0,lookback,pred,train_end,val_end)

    if final_refit: vals=[]
    meta=dict(dataset=f"darts/{name}",train_end=train_end,val_end=val_end,
              total=n,downsample=2 if name=="HeartRate" else (4 if name=="Sunspots" else 1),
              lookback=lookback,forecast_block=pred,test_horizon=n-val_end,
              protocol="Darts time-ordered 60/20/20; recursive full test",
              normalization="train-only in tuning; train+validation refit",mean=float(mu),std=float(std))
    return PreparedData(WindowDataset([z],trains,lookback,pred),
                        WindowDataset([z],vals,lookback,pred),None,1,meta,series=z)


def _read_m4_csv(path):
    df=pd.read_csv(path)
    if df.shape[1]<2:raise ValueError(f"Bad M4 CSV {path}")
    ids=df.iloc[:,0].astype(str).tolist()
    vals=df.iloc[:,1:].to_numpy(dtype=np.float32)
    result=[]
    for row in vals:

        valid=np.isfinite(row)
        if valid.any():
            first_missing=np.flatnonzero(~valid)
            if len(first_missing) and valid[first_missing[0]:].any():
                raise ValueError('M4 CSV contains interior missing entries; no silent time compression')
        vv=row[valid]
        if len(vv): result.append(vv)
        else:result.append(np.zeros((0,),np.float32))
    return ids,result


def official_origins(i, n, L, P, stride, val_fraction, refit=False):
    if min(L, P, stride) < 1 or not 0 < val_fraction < 1:
        raise ValueError('Invalid window configuration')
    starts = range(0, n-L-P, stride)
    cut = len(starts) if refit else int((1-val_fraction)*len(starts))
    return ([(i, L+j) for j in starts[:cut]], [(i, L+j) for j in starts[cut:]])


def official_segments(i, n, L, P, stride, val_fraction, refit=False):
    count=len(range(0,n-L-P,stride))
    cut=count if refit else int((1-val_fraction)*count)
    def segment(start, stop):
        return (i,L+start*stride,L+(stop-1)*stride+P,stride) if stop>start else None
    return segment(0,cut),segment(cut,count)


def load_m4(root,freq,lookback=None,pred=None,refit=False,val_fraction=None,jumps=None):
    if freq not in M4_HORIZONS:raise ValueError(freq)
    root=Path(root)
    def resolve(kind):
        for path in (root/kind/f"{freq}-{kind.lower()}.csv",root/f"{freq}-{kind.lower()}.csv",root/"Dataset"/kind/f"{freq}-{kind.lower()}.csv"):
            if path.exists():return path
        raise FileNotFoundError(f'M4 official {kind} CSV not found under {root}')
    ids,train_rows=_read_m4_csv(resolve('Train'))
    test_ids,test_rows=_read_m4_csv(resolve('Test'))
    if len(set(ids))!=len(ids) or ids!=test_ids:raise ValueError('M4 IDs must be unique and aligned')
    h=M4_HORIZONS[freq]
    L=lookback or {'Quarterly':64,'Weekly':104,'Daily':45,'Yearly':6,'Monthly':40,'Hourly':96}[freq]
    P=pred or h
    stride=jumps or {'Weekly':3,'Daily':5,'Quarterly':2}.get(freq,1)
    if val_fraction is None:val_fraction=.05 if freq in ('Quarterly','Yearly','Monthly') else .1
    zs=[];stats=[];tr=[];va=[];short=[]
    for i,(a,b) in enumerate(zip(train_rows,test_rows)):
        if len(b)!=h or len(a)==0:raise ValueError(f'Invalid official series: {ids[i]}')
        avg=float(a.mean());sd=float(a.std())
        if not np.isfinite(sd) or sd<=0:raise ValueError(f'Zero or invalid standard deviation: {ids[i]}')
        zs.append(((np.concatenate([a,b])-avg)/sd).astype(np.float32)[:,None])
        stats.append(dict(id=ids[i],mean=avg,std=sd,train_n=len(a)))
        if len(a)<L:short.append(ids[i])
        x,v=official_segments(i,len(a),L,P,stride,val_fraction,refit)
        if x is not None:tr.append(x)
        if v is not None:va.append(v)
    if short:raise ValueError(f'{len(short)} M4 test histories shorter than L={L}; no silent exclusion or padding. Select a supported lookback or document and implement an explicit short-history policy.')
    return PreparedData(ImplicitWindows(zs,tr,L,P),ImplicitWindows(zs,va,L,P),None,1,
        dict(dataset=f'm4/{freq}',official_horizon=h,lookback=L,pred_block=P,validation_fraction=val_fraction,validation='chronological eligible window starts; official executable',train_stride=stride,included=len(zs),excluded=0,stats=stats,normalization='whole original training series'),series=zs)


def _read_asset_tensor(path):
    a=np.load(path,allow_pickle=False).astype(np.float32)
    if a.ndim!=3:raise ValueError(f"Crypto requires 3-D asset tensor; got {a.shape}")
    if a.shape[0]==14 and a.shape[-1]==8:return a
    if a.shape[1]==14 and a.shape[-1]==8:return a.transpose(1,0,2)
    raise ValueError("Expected crypto [14,T,8] or [T,14,8], cannot infer")


def load_crypto(root, lookback=80, pred=15, jumps=100, eval_stride=15, refit=False):
    p=Path(root)
    tr=_read_asset_tensor(p/'train.npy')
    te=_read_asset_tensor(p/'test.npy')
    w_path=p/'weights.npy'
    if not w_path.exists():
        raise FileNotFoundError("Original KNF asset-aligned weights.npy required for WRMSE")
    w=np.load(w_path,allow_pickle=False).astype(np.float32)
    if w.shape!=(14,) or not np.isfinite(w).all() or (w<=0).any():
        raise ValueError("weights.npy must hold 14 finite positive values")


    official_weights=np.array([
        4.30406509320417,6.779921907472252,2.3978952727983707,
        4.406719247264253,3.555348061489413,1.3862943611198906,
        5.8944028342648505,2.079441541679836,1.0986122886681098,
        2.3978952727983707,1.0986122886681098,1.6094379124341005,
        2.079441541679836,1.791759469228055],dtype=np.float32)
    if not np.allclose(w,official_weights,rtol=1e-6,atol=1e-6):
        raise ValueError('Crypto weights.npy disagrees with official KRNO/KNF 14-asset vector; '
                         'check export order and weight values')
    asset_manifest=p/'assets.json'
    if not asset_manifest.exists():
        raise FileNotFoundError("assets.json required: auditable identity and order of 14 cryptocurrencies")
    am=json.loads(asset_manifest.read_text())
    if am.get('asset_ids')!=list(range(14)):
        raise ValueError('assets.json must declare asset_ids [0,...,13] in official '
                         'KRNO/KNF weight order; explicitly reorder data before export')
    if tr.shape[-1]!=8 or te.shape[-1]!=8:raise ValueError("Crypto must contain 8 features")
    if lookback>100 or eval_stride!=15:
        raise ValueError('Official Crypto test requires L<=100 and stride=15')
    _check_finite(tr,'crypto train');_check_finite(te,'crypto test')
    mu=tr.mean(axis=1,keepdims=True);sd=tr.std(axis=1,keepdims=True)
    if (sd<=0).any():raise ValueError('Zero Crypto training standard deviation')
    zn=(tr-mu)/sd;ze=(te-mu)/sd
    series=[np.concatenate([zn[i],ze[i]],0) for i in range(14)]
    tr_o=[];v_o=[];te_o=[]
    for i in range(14):
        x,v=official_segments(i,len(tr[i]),lookback,pred,jumps,.1,refit)
        if x is not None:tr_o.append(x)
        if v is not None:v_o.append(v)
        te_o.append((i,len(tr[i])+100,len(series[i])-1,eval_stride))
    return PreparedData(ImplicitWindows(series,tr_o,lookback,pred),
         ImplicitWindows(series,v_o,lookback,pred),ImplicitWindows(series,te_o,lookback,15),
         8,dict(dataset="crypto",lookback=lookback,horizon=15,forecast_block=pred,train_stride=jumps,
                test_stride=eval_stride,channel="last only",weights=w.tolist(),
                asset_ids=am['asset_ids'],per_asset_mean=mu[:,0,:].tolist(),
                per_asset_std=sd[:,0,:].tolist(),test_start=len(tr[0])),series=series)


def _read_trajectories(path):
    a=np.load(path,allow_pickle=False).astype(np.float32)
    if a.ndim !=3 or a.shape[-1]!=2:raise ValueError("Traj requires [N,T,2] velocity")
    _check_finite(a,str(path))
    return a


def load_trajectory(root,lookback=30,pred=15,refit=False,jumps=2):
    p=Path(root)
    tr=_read_trajectories(p/'train.npy');te=_read_trajectories(p/'test.npy')
    series=[];stats=[]
    for v in list(tr)+list(te):
        mu=v.mean(0);sd=v.std(0)
        if (sd<=0).any():raise ValueError('Zero trajectory standard deviation')
        series.append(((v-mu)/sd).astype(np.float32));stats.append(dict(mean=mu.tolist(),std=sd.tolist()))
    train_o=[];val_o=[];test_o=[]
    for i in range(len(tr)):
        a,b=official_origins(i,len(tr[i]),lookback,pred,jumps,.1,refit)
        train_o.extend(a);val_o.extend(b)
    for j in range(len(te)):
        for o in range(50,300-30,50):
            if o<lookback or o+30>len(te[j]):raise ValueError('Missing official trajectory history/target')
            test_o.append((len(tr)+j,o))
    return PreparedData(WindowDataset(series,train_o,lookback,pred),WindowDataset(series,val_o,lookback,pred),None,2,
        dict(dataset='trajectory',input_type='2-D velocity',lookback=lookback,eval_origins=[50,100,150,200,250],eval_horizon=30,forecast_block=pred,all_test_trajectories=len(te),train_examples=len(tr),test_examples=len(te),stats=stats,train_stride=jumps,normalization='official full per-trajectory train and test statistics; test-future access in normalization disclosed'),series=(series,test_o))


class PreprocessedIrregular(Dataset):
    keys=('x','mask','t','q','y','ymask')
    def __init__(self,path,channels=None):
        with np.load(path,allow_pickle=False) as p:
            self.ids=np.asarray(p['record_ids']).astype(str) if 'record_ids' in p else None
            self.offsets=None
            if 'x_offsets' in p:
                self.offsets=(np.asarray(p['x_offsets'],dtype=np.int64),np.asarray(p['y_offsets'],dtype=np.int64))
            self.a={k:np.asarray(p[k],dtype=np.float32) for k in self.keys}
        self.channels=self.a['x'].shape[-1]
        if channels is not None and self.channels!=channels:raise ValueError('Unexpected channel count')
        if self.offsets is not None:
            xo,yo=self.offsets
            if len(xo)!=len(yo) or any(o[0]!=0 or np.any(np.diff(o)<=0) for o in (xo,yo)):
                raise ValueError('Invalid ragged offsets')
            if xo[-1]!=len(self.a['x']) or yo[-1]!=len(self.a['y']):raise ValueError('Ragged offset bounds')
            self.n=len(xo)-1
        else:
            if self.a['x'].ndim!=3 or self.a['y'].ndim!=3:raise ValueError('Expected dense or packed ragged dataset')
            self.n=len(self.a['x'])
        if self.ids is not None and len(self.ids)!=self.n:raise ValueError('Record ID count mismatch')
        for i in range(self.n):
            r=self.record(i)
            x,m,t,q,y,ym=(r[k] for k in self.keys)
            if x.shape!=(len(t),self.channels) or y.shape!=(len(q),self.channels) or m.shape!=x.shape or ym.shape!=y.shape:
                raise ValueError('Inconsistent irregular sample shapes')
            if len(t)<2 or not len(q) or np.any(np.diff(t)<=0) or np.any(np.diff(q)<=0) or q[0]<=t[-1]:
                raise ValueError('Invalid history/target timestamps; preserve upstream support points')
            for k,v in r.items():_check_finite(v,k)
            for v in (m,ym):
                if not np.isin(v,[0,1]).all():raise ValueError('Masks must be binary')
    def record(self,i):
        if self.offsets is None:return {k:v[i] for k,v in self.a.items()}
        xo,yo=self.offsets
        return {k:v[(xo if k in ('x','mask','t') else yo)[i]:(xo if k in ('x','mask','t') else yo)[i+1]] for k,v in self.a.items()}
    def __len__(self):return self.n
    def __getitem__(self,i):return {k:_f32(v) for k,v in self.record(i).items()}


def load_irregular(root,name,expected_seed=None,expected_missing_rate=None):
    channels={'mimic':96,'ushcn':5,'activity':12,'mujoco':14,'synthetic':None}
    if name not in channels:raise ValueError(name)
    p=Path(root)/name
    manifest=json.loads((p/'manifest.json').read_text())
    import hashlib
    req=('source_repository','source_commit','source_split_hashes','preprocessing_description')
    if any(not manifest.get(key) for key in req):raise ValueError('Missing data provenance')
    hashes=manifest['source_split_hashes']
    for fname in ('train.npz','val.npz','test.npz'):
        if hashes.get(fname)!=hashlib.sha256((p/fname).read_bytes()).hexdigest():raise ValueError(f'Split hash mismatch: {fname}')
    tr,va,te=[PreprocessedIrregular(p/f'{k}.npz',channels[name]) for k in ('train','val','test')]
    if name!='synthetic':
        if any(d.ids is None for d in (tr,va,te)):raise ValueError('Official record IDs required for all splits')
        sets=[set(d.ids) for d in (tr,va,te)]
        if any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2))):raise ValueError('Record IDs overlap between splits')
        if name in ('mimic','ushcn','activity') and (manifest.get('split_seed')!=42 or manifest.get('protocol')!='krno_upstream_collate'):
            raise ValueError('Export original KRNO split and collate with the provided exporter')
        if name=='mujoco':
            if manifest.get('protocol')!='stable_neural_sde_krno':raise ValueError('Exact upstream MuJoCo preprocessing export required; independent random masks are not equivalent')
            if expected_seed is None or expected_missing_rate not in (0.,.3,.5,.7):raise ValueError('Supply seed and missingness from the original preprocessing run')
            if manifest.get('seed')!=expected_seed or manifest.get('missing_rate')!=expected_missing_rate:raise ValueError('Wrong MuJoCo preprocessing seed or missingness condition')
            if not manifest.get('missingness_source') or not manifest.get('preprocessing_source_sha256'):raise ValueError('Record original missingness implementation and preprocessing source hash')
            if manifest.get('representation')!='observed_values_masks_timestamps':raise ValueError('Do not substitute interpolated spline values for observations')

    return PreparedData(tr,va,te,tr.channels,dict(dataset=name,source=manifest,variable_length=any(d.offsets is not None for d in (tr,va,te)) or name!='synthetic',protocol='Upstream processed values, masks and timestamps; provenance requires independent raw-data verification'))


class SpatialWindows(Dataset):
    def __init__(self,data,windows):
        self.data=data;self.windows=windows
    def __len__(self):return len(self.windows)
    def __getitem__(self,idx):
        i,o=self.windows[idx]
        x=self.data[i,o-5:o];y=self.data[i,o:o+5]
        return dict(x=_f32(x),mask=torch.ones_like(_f32(x)),
             t=torch.arange(o-5,o,dtype=torch.float32)*.01,
             q=torch.arange(o,o+5,dtype=torch.float32)*.01,
             y=_f32(y),ymask=torch.ones_like(_f32(y)))


def load_shallow(root):
    p=Path(root)
    mf=p/'manifest.json'
    if not mf.exists():
        raise FileNotFoundError('Shallow water requires manifest.json provenance for LOCA split')
    manifest=json.loads(mf.read_text())
    if not manifest.get('source_repository') or not manifest.get('source_commit'):
        raise ValueError('Shallow water manifest requires source_repository/source_commit')
    import hashlib
    source_path=p/'shallow_water.npz'
    expected=manifest.get('sha256')
    if not expected or hashlib.sha256(source_path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Shallow Water needs matching manifest sha256 for original split archive')
    raw=np.load(source_path,allow_pickle=False)
    for key in ('train','test'):
        if key not in raw:raise ValueError('Shallow archive must contain original full LOCA train and test trajectories')
    if 'val' in raw:raise ValueError('Re-export original unsplit LOCA train trajectories; validation is split after windowing')
    def chk(a):
        a=np.asarray(a,dtype=np.float32)
        if a.ndim!=5 or a.shape[2:]!=(32,32,3) or a.shape[1]<60:raise ValueError('Expected [N,T>=60,32,32,3]')
        _check_finite(a,'shallow')
        return a
    a,c=[chk(raw[k]) for k in ('train','test')]
    if len(a)!=1000 or len(c)!=1000:raise ValueError('Official LOCA requires 1000 train and 1000 test trajectories')
    windows=SpatialWindows(a,[(i,o) for i in range(len(a)) for o in range(5,len(a[i])-4)])
    tr,va=torch.utils.data.random_split(windows,[.8,.2],generator=torch.Generator().manual_seed(42))
    return PreparedData(tr,va,None,3,dict(dataset='shallow',physics_channels=['rho','u','v'],context=5,block=5,recursive_steps=55,total_steps_including_context=60,spacing_seconds=.01,spatial_grid=[32,32],train_stride=1,split_seed=42,provenance=manifest),series=c)

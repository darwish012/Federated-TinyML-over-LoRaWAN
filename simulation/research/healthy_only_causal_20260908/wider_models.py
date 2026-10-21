"""Causal wider healthy-only autoencoder ablation (NumPy, no data loading)."""
from __future__ import annotations
import numpy as np

VARIANTS = ("global_h12", "personalized_h12", "svdd_h12")
ROUNDS, EPOCHS, LR, L2 = 12, 2, .03, .0001
ETA, B1, B2, TAU = .01, .9, .99, .001

def _init(seed=17070, hidden=12):
    r=np.random.default_rng(int(seed)); s=np.sqrt(2./(22+hidden))
    return (r.normal(0,s,(22,hidden)), np.zeros(hidden), r.normal(0,s,(hidden,22)), np.zeros(22))
def _fwd(x,a):
    w1,b1,w2,b2=a; h=np.tanh(x@w1+b1); return h@w2+b2,h
def _loss_grads(x,a,c=None):
    w1,b1,w2,b2=a; y,h=_fwd(x,a); n=max(x.size,1); dy=2*(y-x)/n; dh=dy@w2.T
    extra=0
    if c is not None:
        d=h-c; extra=.1*np.mean(d*d); dh += .2*d/max(h.size,1)
    dz=dh*(1-h*h)
    g=(x.T@dz+L2*w1, np.sum(dz,0), h.T@dy+L2*w2, np.sum(dy,0))
    return float(np.mean((y-x)**2)+extra+.5*L2*(np.sum(w1*w1)+np.sum(w2*w2))),g
def _step(a,x,c=None):
    _,g=_loss_grads(x,a,c); return tuple(v-LR*q for v,q in zip(a,g))
def _adam(old, local, m, v, t):
    d=tuple(np.mean(np.stack([q[i]-old[i] for q in local]),0) for i in range(len(old)))
    m=tuple(B1*m[i]+(1-B1)*d[i] for i in range(len(old))); v=tuple(B2*v[i]+(1-B2)*d[i]*d[i] for i in range(len(old)))
    new=tuple(old[i]+ETA*(m[i]/(1-B1**t))/(np.sqrt(v[i]/(1-B2**t))+TAU) for i in range(len(old)))
    return new,m,v
def _center(x,a):
    c=np.mean(_fwd(x,a)[1],0); return np.where(np.abs(c)<.1,np.where(c<0,-.1,.1),c)

def fit(clients:dict[str,np.ndarray],variant:str,seed:int=17070):
    if variant not in VARIANTS: raise ValueError(variant)
    if not clients: raise ValueError("clients must be non-empty")
    xs={k:np.asarray(v,dtype=float) for k,v in clients.items()}
    if any(v.ndim!=2 or v.shape[1]!=22 or v.shape[0] == 0 or not np.all(np.isfinite(v)) for v in xs.values()): raise ValueError("clients must be non-empty finite rows x 22")
    base=_init(seed); centers={k:_center(v,base) for k,v in xs.items()} if variant=="svdd_h12" else {}
    private={k:base for k in xs}; m=tuple(np.zeros_like(v) for v in base); vv=tuple(np.zeros_like(v) for v in base)
    if variant=="personalized_h12":
        em=tuple(np.zeros_like(base[i]) for i in (0,1)); ev=tuple(np.zeros_like(v) for v in em)
    for t in range(1,ROUNDS+1):
        local=[]
        for k,x in xs.items():
            a=(base[0],base[1],private[k][2],private[k][3]) if variant=="personalized_h12" else base
            for _ in range(EPOCHS): a=_step(a,x,centers.get(k))
            local.append(a)
            if variant=="personalized_h12": private[k]=a
        if variant=="personalized_h12":
            d=tuple(np.mean(np.stack([q[i]-base[i] for q in local]),0) for i in (0,1))
            em=tuple(B1*em[i]+(1-B1)*d[i] for i in (0,1)); ev=tuple(B2*ev[i]+(1-B2)*d[i]*d[i] for i in (0,1))
            upd=tuple(base[i]+ETA*(em[i]/(1-B1**t))/(np.sqrt(ev[i]/(1-B2**t))+TAU) for i in (0,1))
            base=(upd[0],upd[1],base[2],base[3])
        else: base,m,vv=_adam(base,local,m,vv,t)
    if variant=="personalized_h12": private={k:(base[0],base[1],a[2],a[3]) for k,a in private.items()}
    return {"variant":variant,"seed":int(seed),"clients":tuple(xs),"global":base,"private":private,"centers":centers}

def _model(s,mid): return s["private"].get(mid,s["global"]) if s["variant"]=="personalized_h12" else s["global"]
def reconstruction_scores(s,mid,x):
    x=np.asarray(x,float); y,_=_fwd(x,_model(s,mid)); return np.mean((y-x)**2,1)
def score(s,mid,x):
    x=np.asarray(x,float); a=_model(s,mid); y,h=_fwd(x,a); z=np.mean((y-x)**2,1)
    if s["variant"]=="svdd_h12": z+=.1*np.mean((h-s["centers"][mid])**2,1)
    return z
def global_arrays(s): return tuple(np.asarray(v) for v in s["global"])
def communication(s):
    n=276 if s["variant"]=="personalized_h12" else 562
    return {"float32_parameters":n,"parameter_bytes":4*n,"header_bytes":16,"fragment_body_bytes":200,"fragments_per_update":int(np.ceil((4*n+16)/200)),"rounds":12,"private_decoder_parameters":286 if s["variant"]=="personalized_h12" else 0}

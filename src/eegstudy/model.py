"""Frozen graph/temporal attention and trained logistic readout. Not an end-to-end GAT."""
import numpy as np
from scipy import signal
from scipy.special import softmax,expit
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score,confusion_matrix
BANDS=[(1,4),(4,8),(8,13),(13,30),(30,40)]
EPS=1e-8
def corrupt(w,kind,seed):
    r=np.random.default_rng(seed); out=w.copy(); c=w.shape[1]
    if kind=='clean': return out
    if kind.startswith('drop'):
        k=int(kind[4:]); out[:,r.choice(c,k,replace=False)]=0
    elif kind=='frontal': out[:,:min(4,c)]=0
    elif kind=='noise':
        ix=r.choice(c,4,replace=False)
        scale=np.std(out[:,ix],axis=-1,keepdims=True)
        out[:,ix]+=r.normal(size=out[:,ix].shape)*scale*4
    else: raise ValueError(kind)
    return out

def features(w,fs=128):
    if w.ndim!=3 or w.shape[1]<4 or w.shape[2]!=int(4*fs) or not np.isfinite(w).all():
        raise ValueError('Expected finite [epochs,channels,4*fs]')
    rms=np.sqrt(np.mean(w*w,axis=-1)); valid=rms>EPS
    med=np.array([np.median(v[m]) if m.any() else 1 for v,m in zip(rms,valid)])
    q=valid*np.exp(-np.abs(np.log((rms+EPS)/(med[:,None]+EPS))))
    f,p=signal.welch(w,fs=fs,nperseg=int(2*fs),noverlap=int(fs),axis=-1)
    bp=np.stack([p[..., (f>=a)&(f<b)].sum(-1) for a,b in BANDS],-1)
    rel=bp/(bp.sum(-1,keepdims=True)+EPS)
    ll=np.mean(np.abs(np.diff(w,axis=-1)),axis=-1)/(rms+EPS)
    z=np.concatenate([np.log(rel+1e-5),np.log(rms[...,None]+EPS),ll[...,None]],-1)
    z[~valid]=0
    def corr(v):
        v=v-v.mean(-1,keepdims=True)
        v=v/(np.sqrt(np.sum(v*v,axis=-1,keepdims=True))+EPS)
        return np.clip(v@v.transpose(0,2,1),-1,1)
    a=np.abs(corr(w)); stability=np.exp(-np.abs(corr(w[...,:w.shape[-1]//2])-corr(w[...,w.shape[-1]//2:])))
    return z,q,a,stability

class Reservoir:
    """Frozen width-24 graph + temporal self-attention; learned readout external."""
    def __init__(self,quality=True,stability=True,temporal=True,seed=17,fs=128):
        self.quality,self.stability,self.temporal=quality,stability,temporal
        r=np.random.default_rng(seed); d=24
        self.W=r.normal(0,1/np.sqrt(7),(7,d))
        self.V=r.normal(0,1/np.sqrt(d),(d,d))
        self.F=r.normal(0,1/np.sqrt(d),(d,d))
        self.Q=r.normal(0,1/np.sqrt(d),(d,d))
        self.d=d; self.fs=fs
    def fit(self,records):
        z=np.concatenate([features(w,self.fs)[0][features(w,self.fs)[1]>0] for w in records])
        if not len(z): raise ValueError("No valid training nodes")
        self.mu=z.mean(0); self.sd=np.maximum(z.std(0),.1)
        return self
    def encode(self,w):
        z,q,a,st=features(w,self.fs); valid=q>0
        z=np.clip((z-self.mu)/self.sd,-5,5); z[~valid]=0
        q=q if self.quality else valid.astype(float)
        if np.any(q.sum(1)<EPS): raise ValueError('All channels invalid in an epoch; reacquire')
        h=np.tanh(z@self.W)
        prior=(a if not self.stability else a*st)+np.eye(w.shape[1])[None]
        logits=(h@h.transpose(0,2,1))/np.sqrt(self.d)+np.log(prior+1e-4)+np.log(q[:,None,:]+EPS)
        logits=np.where(valid[:,None,:],logits,-1e9)
        att=softmax(logits,axis=-1)
        h=np.tanh(h+att@(h@self.V)); h=np.tanh(h+h@self.F)
        u=(h*q[...,None]).sum(1)/q.sum(1)[:,None]
        raw=(z*q[...,None]).sum(1)/q.sum(1)[:,None]
        return u,raw,q.mean(1)
    def pool(self,u,raw):
        if self.temporal:
            j=np.arange(len(u)); l=(u@self.Q)@(u@self.Q).T/np.sqrt(self.d)-np.abs(j[:,None]-j[None,:])/4
            v=np.tanh(u+softmax(l,axis=1)@u)
        else: v=u
        return np.r_[v.mean(0),raw.mean(0),raw.std(0)]
    def transform(self,records):
        return np.stack([self.pool(*self.encode(w)[:2]) for w in records])
    def prefixes(self,w,steps):
        u,raw,q=self.encode(w)
        return np.stack([self.pool(u[:k],raw[:k]) for k in steps])

class Readout:
    def fit(self,x,y,weight=None):
        self.scale=StandardScaler().fit(x)
        self.clf=LogisticRegression(C=.1,max_iter=1000,solver='lbfgs').fit(self.scale.transform(x),y,sample_weight=weight)
        return self
    def logits(self,x): return self.clf.decision_function(self.scale.transform(x))
    def prob(self,x,temp=1): return expit(self.logits(x)/temp)

def metrics(y,p):
    pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(balanced_accuracy=float(balanced_accuracy_score(y,pred)),auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
                f1=float(f1_score(y,pred,zero_division=0)),sensitivity=float(tp/max(tp+fn,1)),
                specificity=float(tn/max(tn+fp,1)),brier=float(np.mean((p-y)**2)))

def temperature(logits,y):
    def loss(logt):
        z=logits/np.exp(logt)
        return np.mean(np.logaddexp(0,z)-y*z)
    return float(np.exp(minimize_scalar(loss,bounds=(-2,2),method='bounded').x))

def stop_policy(p,steps,threshold,drift=.08):
    """Stable class AND confidence at two consecutive checkpoints. No guarantee."""
    for j in range(1,len(steps)-1):
        if (p[j]>=.5)==(p[j-1]>=.5) and min(max(p[j],1-p[j]),max(p[j-1],1-p[j-1]))>=threshold and abs(p[j]-p[j-1])<=drift:
            return j
    return len(steps)-1


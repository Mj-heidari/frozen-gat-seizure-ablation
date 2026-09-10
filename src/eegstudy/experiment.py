"""Patient/family-disjoint evaluation with independently fitted preprocessing."""
from pathlib import Path
import csv,json,pickle,platform,time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from .model import Reservoir,Readout,features,corrupt,metrics,temperature,stop_policy
from .data import write_csv

def group_folds(y,g,n=5,seed=41):
    if len(np.unique(g))<n*2:raise ValueError('Too few independent groups for requested folds')
    for tr,te in StratifiedGroupKFold(n,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,g):
        if set(g[tr])&set(g[te]):raise AssertionError('Group leakage')
        if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2:raise ValueError('A fold lacks a class; revise cohort/fold plan before evaluation')
        yield tr,te

def weights(y,g):
    # Equal total mass per group first, then balance aggregate class mass.
    w=np.array([1/np.sum(g==v) for v in g],float)
    for cls in (0,1):
        mass=w[y==cls].sum()
        if mass==0:raise ValueError('Training requires both classes')
        w[y==cls]/=mass
    return w*len(w)/w.sum()

def ba(y,p):
    return .5*(np.mean(p[y==1]>=.5)+np.mean(p[y==0]<.5))

def cluster_ci(y,p,g,q=None,seed=900,repeats=1000):
    """Percentile bootstrap of whole patient/family groups; descriptive OOF interval."""
    rng=np.random.default_rng(seed);unique=np.unique(g);ixs={k:np.flatnonzero(g==k) for k in unique};scores=[]
    for _ in range(repeats):
        ix=np.concatenate([ixs[k] for k in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[ix]))<2:continue
        score=ba(y[ix],p[ix]);scores.append(score if q is None else score-ba(y[ix],q[ix]))
    if not scores:return None
    return np.quantile(scores,[.025,.975]).tolist()

def chb_run(data,y,g,ids,cfg,out):
    variants={'spectral':(False,False,False),'plain':(False,False,False),'quality':(True,False,False),'proposed':(True,True,False),'proposed_aug':(True,True,True)}
    conditions=['clean','drop2','drop4','noise'];pred={v:{c:np.zeros(len(y)) for c in conditions} for v in variants};splits=[]
    for fold,(tr,te) in enumerate(group_folds(y,g,cfg['folds'],cfg['seed'])):
        splits.append({'fold':fold,'train_groups':sorted(set(g[tr])),'test_groups':sorted(set(g[te]))})
        for name,(quality,stable,augment) in variants.items():
            net=Reservoir(quality,stable,False,fs=cfg['fs']).fit([data[i] for i in tr])
            train=[];yy=[];gg=[]
            for i in tr:
                for mode in (['clean','drop2','noise'] if augment else ['clean']):
                    train.append(corrupt(data[i],mode,cfg['seed']+int(i)));yy.append(y[i]);gg.append(g[i])
            def representation(records):
                z=net.transform(records);return z[:,24:31] if name=='spectral' else z[:,:31]
            head=Readout().fit(representation(train),np.array(yy),weights(np.array(yy),np.array(gg)))
            for c in conditions:
                xx=representation([corrupt(data[i],c,70000+int(i)) for i in te]);pred[name][c][te]=head.prob(xx)
            with open(out/f'fold{fold}_{name}.pkl','wb') as f:pickle.dump({'net':net,'head':head,'study':'chb','model':name,'fs':cfg['fs']},f)
    summary={v:{c:{**metrics(y,p),'ba_group_ci95':cluster_ci(y,p,g)} for c,p in vv.items()} for v,vv in pred.items()}
    differences={c:{'delta_ba':float(ba(y,pred['proposed'][c])-ba(y,pred['plain'][c])),'ci95':cluster_ci(y,pred['proposed'][c],g,pred['plain'][c])} for c in conditions}
    rows=[{'sample':str(ids[i]),'group':str(g[i]),'label':int(y[i]),'model':v,'condition':c,'probability':float(p[i])} for v,vv in pred.items() for c,p in vv.items() for i in range(len(y))]
    return summary,rows,splits,{'paired_proposed_minus_plain':differences}

def nested_roles(tr,y,g,seed):
    fit_rel,val_rel=next(StratifiedGroupKFold(4,shuffle=True,random_state=seed).split(np.zeros(len(tr)),y[tr],g[tr]))
    fit,val=tr[fit_rel],tr[val_rel]
    c,p=next(StratifiedGroupKFold(2,shuffle=True,random_state=seed+1).split(np.zeros(len(val)),y[val],g[val]))
    cal,policy=val[c],val[p]
    for ix in (fit,cal,policy):
        if len(np.unique(y[ix]))!=2:raise ValueError('Nested role lacks a class; add independent subjects or predeclare different folds')
    return fit,cal,policy

def hbn_run(data,y,g,ids,cfg,out):
    K=cfg['max_epochs_hbn'];steps=list(range(2,K+1,2))
    if len(steps)<3 or K%2:raise ValueError('HBN budget must be an even number >=6 epochs')
    if min(map(len,data))<K:raise ValueError('Insufficient recording length')
    probs=np.zeros((len(y),len(steps)));stop=np.zeros(len(y),int);conf=np.zeros(len(y),int);base=np.zeros(len(y));splits=[];policies=[]
    for fold,(tr,te) in enumerate(group_folds(y,g,cfg['folds'],cfg['seed'])):
        fit,cal,policy=nested_roles(tr,y,g,cfg['seed']+fold)
        roles={'fit':fit,'temperature':cal,'policy':policy,'test':te}
        for a,ia in roles.items():
            for b,ib in roles.items():
                if a!=b and set(g[ia])&set(g[ib]):raise AssertionError('Nested group leakage')
        splits.append({'fold':fold,**{k:sorted(set(g[v])) for k,v in roles.items()}})
        net=Reservoir(fs=cfg['fs']).fit([data[i][:K] for i in fit])
        xx=np.concatenate([net.prefixes(data[i][:K],steps) for i in fit]);yy=np.repeat(y[fit],len(steps));gg=np.repeat(g[fit],len(steps))
        head=Readout().fit(xx,yy,weights(yy,gg))
        # An explicit spectral readout receives the same full recording budget.
        bx=net.transform([data[i][:K] for i in fit])[:,24:]
        bhead=Readout().fit(bx,y[fit],weights(y[fit],g[fit]))
        base[te]=bhead.prob(net.transform([data[i][:K] for i in te])[:,24:])
        xc=np.concatenate([net.prefixes(data[i][:K],steps) for i in cal]);temp=temperature(head.logits(xc),np.repeat(y[cal],len(steps)))
        pp=np.stack([head.prob(net.prefixes(data[i][:K],steps),temp) for i in policy]);fullba=ba(y[policy],pp[:,-1]);candidates=[]
        for tau in [.75,.85,.9,.95,.99,1.01]:
            j=np.array([stop_policy(p,steps,tau) for p in pp]);b=ba(y[policy],pp[np.arange(len(j)),j])
            if b>=fullba-.02:candidates.append((float(np.mean(np.array(steps)[j])),tau))
        tau=min(candidates)[1]
        for i in te:
            p=head.prob(net.prefixes(data[i][:K],steps),temp);probs[i]=p;stop[i]=stop_policy(p,steps,tau)
            eligible=np.flatnonzero(np.maximum(p,1-p)>=tau);conf[i]=eligible[0] if len(eligible) else len(steps)-1
        policies.append({'fold':fold,'temperature':temp,'threshold':tau,'policy_n':len(policy),'policy_full_ba':float(fullba)})
        with open(out/f'fold{fold}.pkl','wb') as f:pickle.dump({'net':net,'head':head,'study':'hbn','fs':cfg['fs'],'temperature':temp,'threshold':tau,'steps':steps,'spectral_head':bhead},f)
    modes={'full':np.full(len(y),len(steps)-1),'first':np.zeros(len(y),int),'confidence_only':conf,'stable_stop':stop};summary={}
    for mode,j in modes.items():
        p=probs[np.arange(len(y)),j];sec=np.array(steps)[j]*4
        summary[mode]={**metrics(y,p),'ba_group_ci95':cluster_ci(y,p,g),'mean_seconds':float(sec.mean()),'early_fraction':float(np.mean(j<len(steps)-1))}
    summary['spectral_full']={**metrics(y,base),'ba_group_ci95':cluster_ci(y,base,g),'mean_seconds':K*4,'early_fraction':0.0}
    sp=probs[np.arange(len(y)),stop]
    rows=[{'subject':str(ids[i]),'group':str(g[i]),'label':int(y[i]),'checkpoint_seconds':steps[j]*4,'probability':float(probs[i,j]),'stopped':int(j==stop[i])} for i in range(len(y)) for j in range(len(steps))]
    extra={'policies':policies,'steps':steps,'paired_stable_minus_full':{'delta_ba':float(ba(y,sp)-ba(y,probs[:,-1])),'ci95':cluster_ci(y,sp,g,probs[:,-1])}}
    return summary,rows,splits,extra

def run(data,y,g,ids,cfg,out,kind):
    import scipy,sklearn
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be new/empty')
    if len(set(ids))!=len(ids):raise ValueError('Duplicate sample/subject identifiers')
    if set(np.unique(y))!={0,1}:raise ValueError('Binary labels 0 and 1 required')
    out.mkdir(parents=True,exist_ok=True);t=time.perf_counter()
    summary,rows,splits,extra=(chb_run if cfg['study']=='chb' else hbn_run)(data,y,g,ids,cfg,out)
    result={'data_kind':kind,'clinical_validation':False,'study':cfg['study'],'n_samples':len(y),'n_groups':len(set(g)),'seconds':time.perf_counter()-t,'summary':summary,'config':cfg,'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},**extra}
    (out/'results.json').write_text(json.dumps(result,indent=2));(out/'splits.json').write_text(json.dumps(splits,indent=2));write_csv(out/'predictions.csv',rows)
    print(json.dumps({k:result[k] for k in ['data_kind','study','n_samples','n_groups','seconds']}))
    return result

def demo(cfg):
    """Label-dependent oscillator fixture; no clinical realism is asserted."""
    rng=np.random.default_rng(2026);fs=cfg['fs'];t=np.arange(fs*4)/fs
    n=12 if cfg['study']=='chb' else 80;C=18 if cfg['study']=='chb' else 16
    data=[];ys=[];gs=[];ids=[]
    for s in range(n):
        subject_gain=np.exp(rng.normal(0,.3));alpha_offset=rng.normal(0,.3);theta_offset=rng.normal(0,.3);ws=[]
        for j in range(12):
            y=int(j>=8) if cfg['study']=='chb' else s%2
            phase=rng.uniform(0,6,(C,1));trait=rng.normal(0,.15)
            x=subject_gain*(np.exp(.1-.25*y+trait+alpha_offset+rng.normal(0,.25))*np.sin(2*np.pi*(10+rng.normal(0,.5))*t+phase)+np.exp(-.2+.3*y+trait+theta_offset+rng.normal(0,.25))*np.sin(2*np.pi*(6+rng.normal(0,.5))*t+phase*.7)+rng.normal(0,.8,(C,len(t))))
            ws.append(x.astype('float32'))
            if cfg['study']=='chb':data.append(np.array(ws[-1:]));ys.append(y);gs.append(f'sim-{s:03d}');ids.append(f'sim-{s:03d}:{j}')
        if cfg['study']=='hbn':data.append(np.stack(ws));ys.append(s%2);gs.append(f'sim-{s:03d}');ids.append(f'sim-{s:03d}')
    return data,np.array(ys),np.array(gs),np.array(ids)

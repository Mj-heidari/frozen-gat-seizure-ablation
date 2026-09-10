"""Explicit manifests: no inferred clinical labels, no random epoch splitting."""
from pathlib import Path
from fractions import Fraction
import csv, hashlib, json, re
import numpy as np
from scipy import signal, io

FORMATS={'.edf','.bdf','.set','.fif','.npz'}

def write_csv(path,rows,fields=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def rows_csv(path,delimiter=','):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f,delimiter=delimiter))

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def patient_group(subject):
    # BIDS conversion may rename subjects; explicit group is still required.
    m=re.fullmatch(r'(?:sub-)?chb0*(\d+)',subject,re.I)
    if m:
        n=int(m.group(1));return 'chb01' if n==21 else f'chb{n:02d}'
    return subject

def safe_child(root,rel):
    root=Path(root).resolve();p=(root/rel).resolve()
    if not p.is_relative_to(root):raise ValueError('Manifest paths must stay inside dataset root')
    return p

def inventory(cfg,out):
    root=Path(cfg['root']);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not root.is_dir():raise ValueError(f'Dataset root not accessible: {root}')
    rows=[]
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() not in FORMATS or not p.is_file():continue
        rel=p.relative_to(root).as_posix()
        m=re.search(r'(sub-[A-Za-z0-9]+|chb\d+)',rel,re.I)
        sid=m.group(1) if m else ''
        events=p.with_name(re.sub(r'_eeg$', '_events',p.stem)+'.tsv')
        rows.append(dict(include=0,path=rel,subject=sid,group=patient_group(sid),start_s=0,stop_s='',events=events.relative_to(root).as_posix() if events.exists() else '',annotation_complete=0,task='',bytes=p.stat().st_size))
    if not rows:raise ValueError('No supported recordings found; see docs/INPUT_FORMATS.md for MAT conversion')
    write_csv(out/'manifest.csv',rows)
    (out/'inventory.json').write_text(json.dumps({'recordings':len(rows),'formats':sorted({Path(r['path']).suffix for r in rows}),'note':'All include=0 until reviewed. No diagnostic labels inferred.'},indent=2))
    write_csv(out/'labels.example.csv',[{'subject':'REPLACE_WITH_EXACT_ID','label':'','group':'','label_source':'','task':''}])
    print(f'{len(rows)} recordings indexed. Review {out / "manifest.csv"}.')

def inspect_record(path):
    p=Path(path)
    if p.suffix=='.npz':
        with np.load(p,allow_pickle=False) as z:
            return {'channels':z['channels'].astype(str).tolist(),'fs':float(z['fs']),'shape':list(z['data'].shape),'unit':str(z['unit'])}
    raw=_mne_raw(p)
    result={'channels':raw.ch_names,'fs':raw.info['sfreq'],'samples':raw.n_times,'seconds':raw.n_times/raw.info['sfreq'],'unit':'V (MNE output)'}
    raw.close();return result

def _mne_raw(p):
    try:import mne
    except ImportError as e:raise RuntimeError('Install reader dependencies: python -m pip install -e ".[io]"') from e
    readers={'.edf':mne.io.read_raw_edf,'.bdf':mne.io.read_raw_bdf,'.set':mne.io.read_raw_eeglab,'.fif':mne.io.read_raw_fif}
    if p.suffix.lower() not in readers:raise ValueError('Unsupported format; use documented NPZ interchange')
    return readers[p.suffix.lower()](p,preload=False,verbose='ERROR')

def read_segment(path,start_s,stop_s):
    """Return channels x samples, fs, names, in microvolts. Read only selected range."""
    p=Path(path)
    if start_s<0 or stop_s<=start_s:raise ValueError('Explicit start_s < stop_s required')
    if p.suffix.lower()=='.npz':
        with np.load(p,allow_pickle=False) as z:
            fs=float(z['fs']); names=z['channels'].astype(str).tolist(); x=z['data'];unit=str(z['unit'])
            if unit not in ('uV','V'):raise ValueError('NPZ unit must be V or uV')
            a,b=int(round(start_s*fs)),int(round(stop_s*fs))
            if x.ndim!=2 or x.shape[0]!=len(names) or b>x.shape[1]:raise ValueError('NPZ shape or interval mismatch')
            return np.array(x[:,a:b],dtype=float)*(1e6 if unit=='V' else 1),fs,names
    raw=_mne_raw(p)
    try:
        fs=float(raw.info['sfreq']);a,b=int(round(start_s*fs)),int(round(stop_s*fs))
        if b>raw.n_times:raise ValueError('stop_s exceeds recording duration')
        return raw.get_data(start=a,stop=b)*1e6,fs,list(raw.ch_names)
    finally:raw.close()

def canonicalize(x,names,cfg):
    target=cfg['channels'];aliases=cfg.get('channel_aliases',{})
    if len(target)<4 or len({v.upper() for v in target})!=len(target):raise ValueError('Configure >=4 distinct channels after header inspection')
    mapped=[aliases.get(n,n).upper() for n in names]
    if any(mapped.count(c.upper())>1 for c in target):raise ValueError('Ambiguous duplicate target channel names; review the source')
    missing=[c for c in target if c.upper() not in mapped]
    if missing and not cfg.get('allow_missing_channels',False):raise ValueError(f'Missing configured channels: {missing}')
    out=np.zeros((len(target),x.shape[1]),dtype=float)
    for j,c in enumerate(target):
        if c.upper() in mapped:out[j]=x[mapped.index(c.upper())]
    if not np.isfinite(out).all():raise ValueError('Nonfinite samples; repair input explicitly')
    return out,missing

def epochs(x,fs,target_fs):
    """Four-second blocks, local zero-phase filter, polyphase resample. No rereference."""
    if fs<100 or target_fs not in (100,128):raise ValueError('Expected input >=100 Hz, target 100 or 128 Hz')
    n=round(4*fs);k=x.shape[1]//n
    if not k:raise ValueError('Fewer than four seconds')
    ratio=Fraction(target_fs/fs).limit_denominator(10000)
    sos=signal.butter(4,[1,40],fs=fs,btype='bandpass',output='sos')
    out=[]
    for j in range(k):
        w=x[:,j*n:(j+1)*n];w=w-w.mean(-1,keepdims=True)
        w=signal.sosfiltfilt(sos,w,axis=-1)
        w=signal.resample_poly(w,ratio.numerator,ratio.denominator,axis=-1)
        if w.shape[-1]!=4*target_fs:raise ValueError('Sample-rate rounding mismatch')
        out.append(w.astype('float32'))
    return np.stack(out)

def seizure_intervals(path,cfg,recording_end):
    if not path:raise ValueError('CHB annotation file is required even for seizure-free recordings')
    rows=rows_csv(path,'\t');intervals=[]
    # An empty table is allowed only because manifest annotation_complete must be 1.
    for r in rows:
        if cfg['seizure_column'] not in r:raise ValueError('Configured seizure column absent')
        if r[cfg['seizure_column']] not in cfg['seizure_values']:continue
        a=float(r['onset']);d=float(r['duration']);b=a+d
        if not np.isfinite([a,b]).all() or a<0 or d<=0:raise ValueError('Invalid seizure interval')
        intervals.append((a,b))
    intervals.sort()
    for j in range(1,len(intervals)):
        if intervals[j][0]<intervals[j-1][1]:raise ValueError('Overlapping seizure annotations need review')
    return intervals

def window_labels(starts,intervals,width=4):
    labels=[]
    for a in starts:
        b=a+width
        if any(a>=s and b<=e for s,e in intervals):labels.append(1)
        elif any(a<e and b>s for s,e in intervals):labels.append(-1)
        else:labels.append(0)
    return np.array(labels)

def prepare(cfg):
    root=Path(cfg['root']);out=Path(cfg['cache'])
    if out.exists() and any(out.iterdir()):raise ValueError('Cache must be empty; use a new folder to avoid mixed runs')
    chosen=[r for r in rows_csv(cfg['manifest']) if r['include']=='1']
    if not chosen:raise ValueError('No reviewed include=1 recordings')
    for r in chosen:
        if not r['subject'] or not r['group']:raise ValueError('Explicit subject and group required')
        if not r['stop_s']:raise ValueError('Set an explicit stop_s for every selected recording')
    paths=[safe_child(root,r['path']) for r in chosen]
    for p in set(paths):
        rr=[r for r,pp in zip(chosen,paths) if pp==p]
        if len({r['group'] for r in rr})!=1:raise ValueError('One source file assigned to multiple groups')
        intervals=sorted((float(r['start_s']),float(r['stop_s'])) for r in rr)
        if any(b[0]<a[1] for a,b in zip(intervals,intervals[1:])):raise ValueError('Overlapping selected intervals would duplicate samples')
    sources=set(paths)
    for p in paths:
        if p.suffix=='.set':sources.update(p.parent.glob('*.fdt'))
    total=sum(p.stat().st_size for p in sources)
    if total>cfg['max_source_bytes']:raise ValueError(f'Selected source bytes {total} exceed budget; review subset')
    labels={}
    if cfg['study']=='hbn':
        if cfg['hbn_task']=='REVIEW_REQUIRED' or not cfg['channels']:raise ValueError('Review HBN task and montage in config')
        for r in rows_csv(cfg['labels']):
            if r['subject'] in labels:raise ValueError('Duplicate subject phenotype')
            if r['label'] not in ('0','1') or not r.get('label_source'):raise ValueError('Verified binary label and source required')
            labels[r['subject']]=r
        if len({r['subject'] for r in chosen})!=len(chosen):raise ValueError('HBN needs one selected recording per subject')
    # Bound the peak raw array, output cache and per-record memory before loading.
    estimate=sum(int((float(r['stop_s'])-float(r['start_s']))/4)*len(cfg['channels'])*4*cfg['fs']*4 for r in chosen)
    if estimate>cfg['max_cache_bytes']:raise ValueError(f'Estimated cache {estimate} exceeds budget')
    out.mkdir(parents=True,exist_ok=True);manifest=[]
    try:
        for k,(r,p) in enumerate(zip(chosen,paths)):
            start,stop=float(r['start_s']),float(r['stop_s'])
            if stop-start>7200:raise ValueError('A selected interval may not exceed two hours')
            x,fs,names=read_segment(p,start,stop);x,missing=canonicalize(x,names,cfg);w=epochs(x,fs,cfg['fs'])
            if cfg['study']=='chb':
                if r['annotation_complete']!='1':raise ValueError('Confirm full seizure annotation coverage first')
                ep=safe_child(root,r['events']) if r['events'] else None
                intervals=seizure_intervals(ep,cfg,stop)
                yy=window_labels(start+4*np.arange(len(w)),intervals);keep=yy>=0
                w,yy=w[keep],yy[keep];times=(start+4*np.arange(len(keep)))[keep]
                if not len(w):raise ValueError('No fully labeled windows')
                group=patient_group(r['group']);source_label=file_hash(ep)
            else:
                sid=r['subject']
                if sid not in labels:raise ValueError(f'Missing phenotype: {sid}')
                lab=labels[sid]
                if r['task']!=cfg['hbn_task'] or lab['task']!=cfg['hbn_task']:raise ValueError('HBN task mismatch')
                if lab['group']!=r['group']:raise ValueError('Family/group metadata mismatch')
                K=cfg['max_epochs_hbn']
                if len(w)<K:raise ValueError('HBN recording shorter than fixed prefix budget')
                w=w[:K];yy=np.full(K,int(lab['label']));times=start+4*np.arange(K)
                group=r['group'];source_label=lab['label_source']
            dest=out/f'record-{k:05d}.npz'
            np.savez_compressed(dest,x=w,y=yy,subject=r['subject'],group=group,fs=cfg['fs'],starts=times,data_kind='real')
            manifest.append({'cache':dest.name,'source':r['path'],'sha256_source':file_hash(p),'sha256_cache':file_hash(dest),'label_source':source_label,'windows':len(w),'positive_windows':int(yy.sum()),'subject':r['subject'],'group':group,'missing_channels':missing,'start_s':start,'stop_s':stop})
        (out/'manifest.json').write_text(json.dumps({'status':'complete','study':cfg['study'],'source_bytes':total,'estimated_array_bytes':estimate,'config':cfg,'records':manifest},indent=2))
    except Exception:
        (out/'INCOMPLETE.txt').write_text('Preparation failed; do not train. Inspect error, delete this generated cache, retry.')
        raise
    print(f'Prepared {len(manifest)} recordings; estimated arrays {estimate/1e6:.1f} MB.')

def load_cache(cfg):
    root=Path(cfg['cache'])
    if (root/'INCOMPLETE.txt').exists():raise ValueError('Incomplete cache')
    meta=json.loads((root/'manifest.json').read_text())
    if meta.get('status')!='complete' or meta['study']!=cfg['study']:raise ValueError('Cache study/status mismatch')
    if meta['config']['fs']!=cfg['fs'] or meta['config']['channels']!=cfg['channels']:raise ValueError('Cache preprocessing config changed')
    data=[];ys=[];groups=[];ids=[]
    for row in meta['records']:
        p=root/row['cache']
        if file_hash(p)!=row['sha256_cache']:raise ValueError('Cache integrity mismatch')
        with np.load(p,allow_pickle=False) as z:
            if str(z['data_kind'])!='real':raise ValueError('Real-data command refuses synthetic cache')
            x=z['x'];y=z['y'];g=str(z['group']);sid=str(z['subject'])
            if cfg['study']=='chb':
                for j in range(len(x)):
                    data.append(x[j:j+1]);ys.append(int(y[j]));groups.append(g);ids.append(f'{row["cache"]}:{j}')
            else:data.append(x);ys.append(int(y[0]));groups.append(g);ids.append(sid)
    return data,np.array(ys),np.array(groups),np.array(ids)

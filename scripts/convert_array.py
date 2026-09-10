"""Explicit MAT-v5/NPY conversion; does not guess layout, units, channels, or labels."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--out',required=True);p.add_argument('--data-key',default='data');p.add_argument('--fs',type=float,required=True);p.add_argument('--channels-json',required=True);p.add_argument('--unit',choices=['V','uV'],required=True);p.add_argument('--layout',choices=['channels_first','samples_first'],required=True);a=p.parse_args()
src=Path(a.source)
if src.suffix=='.npy':x=np.load(src,allow_pickle=False)
else:
    obj=loadmat(src,simplify_cells=True)
    for key in a.data_key.split('.'):obj=obj[key]
    x=np.asarray(obj)
if a.layout=='samples_first':x=x.T
ch=json.loads(Path(a.channels_json).read_text())
if x.ndim!=2 or len(ch)!=x.shape[0] or len(set(ch))!=len(ch) or not np.isfinite(x).all():raise ValueError('Input shape/channel/finite-value contract failed')
if Path(a.out).exists():raise ValueError('Refusing to overwrite existing output')
np.savez_compressed(a.out,data=x.astype('float32'),fs=a.fs,channels=np.array(ch),unit=a.unit)
print(f'Wrote {a.out}: shape={x.shape}, fs={a.fs}, unit={a.unit}')

"""Use python -m eegstudy.cli --help. Real data always require reviewed manifests."""
from pathlib import Path
import argparse,json,pickle
import numpy as np
from . import data,experiment
from .model import stop_policy

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='cmd',required=True)
    for name in ['inventory','prepare','run','demo']:
        p=sub.add_parser(name);p.add_argument('--config',default='configs/windows.json')
        if name in ['inventory','run','demo']:p.add_argument('--out',required=True)
    p=sub.add_parser('inspect');p.add_argument('record')
    p=sub.add_parser('predict');p.add_argument('--checkpoint',required=True);p.add_argument('--record',required=True)
    a=parser.parse_args()
    if a.cmd=='inspect':print(json.dumps(data.inspect_record(a.record),indent=2));return
    if a.cmd=='predict':
        # Only load your own locally generated checkpoints; pickle is executable.
        with open(a.checkpoint,'rb') as f:ck=pickle.load(f)
        with np.load(a.record,allow_pickle=False) as z:
            x=z['x'];fs=float(z['fs'])
        if fs!=ck['fs']:raise ValueError('Checkpoint sample rate mismatch')
        net,head=ck['net'],ck['head']
        if ck['study']=='chb':
            xx=net.transform([w[None] for w in x]);xx=xx[:,24:31] if ck['model']=='spectral' else xx[:,:31]
            print(json.dumps({'window_probabilities':head.prob(xx).tolist()}))
        else:
            steps=ck['steps']
            if len(x)<steps[-1]:raise ValueError('Recording shorter than configured budget')
            p=head.prob(net.prefixes(x[:steps[-1]],steps),ck['temperature']);j=stop_policy(p,steps,ck['threshold'])
            print(json.dumps({'probability':float(p[j]),'seconds':steps[j]*4,'clinical_validation':False}))
        return
    cfg=json.loads(Path(a.config).read_text())
    if a.cmd=='inventory':data.inventory(cfg,a.out)
    elif a.cmd=='prepare':data.prepare(cfg)
    else:
        args=experiment.demo(cfg) if a.cmd=='demo' else data.load_cache(cfg)
        experiment.run(*args,cfg,a.out,'synthetic' if a.cmd=='demo' else 'real')
if __name__=='__main__':main()

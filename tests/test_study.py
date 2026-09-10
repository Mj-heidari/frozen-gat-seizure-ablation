import unittest,tempfile,json
from pathlib import Path
import numpy as np
from scipy.io import savemat
from eegstudy.data import patient_group,safe_child,epochs,canonicalize,window_labels,read_segment,seizure_intervals,write_csv,prepare,load_cache
from eegstudy.model import Reservoir,Readout,corrupt,stop_policy,features
from eegstudy.experiment import group_folds,nested_roles,weights,cluster_ci

class ScientificContracts(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(4);self.x=self.rng.normal(size=(6,8,400)).astype('float32')
    def test_chb_repeated_patient(self):
        self.assertEqual(patient_group('sub-chb21'),patient_group('chb01'))
        self.assertNotEqual(patient_group('chb02'),patient_group('chb01'))
    def test_window_boundaries(self):
        np.testing.assert_array_equal(window_labels([0,4,8,12],[(4,10)]),[0,1,-1,0])
    def test_epoch_does_not_see_future(self):
        x=self.x.transpose(1,0,2).reshape(8,-1)
        a=epochs(x,100,100);x[:,800:]+=100*self.rng.normal(size=x[:,800:].shape)
        b=epochs(x,100,100);np.testing.assert_array_equal(a[:2],b[:2])
    def test_prefix_does_not_see_future(self):
        net=Reservoir(fs=100).fit([self.x]);a=net.prefixes(self.x,[2,4,6]);z=self.x.copy();z[2:]*=10
        b=net.prefixes(z,[2,4,6]);np.testing.assert_allclose(a[0],b[0],atol=1e-12)
    def test_inference_cannot_refit_normalizer(self):
        net=Reservoir(fs=100).fit([self.x]);m=net.mu.copy();s=net.sd.copy();net.transform([self.x*100])
        np.testing.assert_array_equal(net.mu,m);np.testing.assert_array_equal(net.sd,s)
    def test_montage_alias_collision_rejected(self):
        with self.assertRaises(ValueError):canonicalize(np.ones((4,20)),['a','b','c','d'],{'channels':['a','b','c','d'],'channel_aliases':{'b':'a'}})
    def test_missing_channel_requires_explicit_option(self):
        cfg={'channels':['a','b','c','d','e']}
        with self.assertRaises(ValueError):canonicalize(np.ones((4,20)),['a','b','c','d'],cfg)
        cfg['allow_missing_channels']=True;x,m=canonicalize(np.ones((4,20)),['a','b','c','d'],cfg)
        self.assertEqual(m,['e']);self.assertTrue(np.all(x[-1]==0))
    def test_all_flat_rejected(self):
        net=Reservoir(fs=100).fit([self.x])
        with self.assertRaises(ValueError):net.transform([self.x*0])
    def test_corruption_paired_repeatable(self):
        np.testing.assert_array_equal(corrupt(self.x,'drop2',12),corrupt(self.x,'drop2',12))
        self.assertEqual(np.sum(np.all(corrupt(self.x,'drop2',12)==0,axis=(0,2))),2)
    def test_group_split_disjoint(self):
        g=np.repeat([f'p{i}' for i in range(20)],4);y=np.tile([0,0,1,1],20)
        seen=[]
        for tr,te in group_folds(y,g):self.assertFalse(set(g[tr])&set(g[te]));seen.extend(te)
        self.assertEqual(sorted(seen),list(range(len(y))))
    def test_nested_roles_disjoint(self):
        y=np.arange(80)%2;g=np.array([f'p{i}' for i in range(80)])
        for tr,te in group_folds(y,g):
            a,b,c=nested_roles(tr,y,g,41)
            self.assertEqual(len(set(a)|set(b)|set(c)|set(te)),80)
    def test_weights_balance_groups_and_classes(self):
        y=np.array([0,0,0,1]);g=np.array(['a','a','b','c']);w=weights(y,g)
        self.assertAlmostEqual(w[y==0].sum(),w[y==1].sum());self.assertAlmostEqual(w[0]+w[1],w[2])
    def test_stopping_requires_two_stable_checkpoints(self):
        self.assertEqual(stop_policy([.99,.6,.99],[2,4,6],.9),2)
        self.assertEqual(stop_policy([.95,.96,.99],[2,4,6],.9),1)
        self.assertEqual(stop_policy([.95,.96,.99],[2,4,6],1.01),2)
    def test_cluster_difference_zero_for_equal_predictions(self):
        y=np.arange(20)%2;p=.1+.8*y;g=np.arange(20)
        self.assertEqual(cluster_ci(y,p,g,p,repeats=30),[0.,0.])
    def test_npz_units_and_range(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.npz';np.savez(p,data=np.ones((4,800))*1e-6,fs=100,channels=['a','b','c','d'],unit='V')
            x,fs,ch=read_segment(p,0,4);self.assertEqual(x.shape,(4,400));np.testing.assert_allclose(x,1)
            with self.assertRaises(ValueError):read_segment(p,0,10)
    def test_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):safe_child(d,'../outside')
    def test_prepare_cache_integrity_and_annotation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'eeg.npz';np.savez(p,data=self.x.transpose(1,0,2).reshape(8,-1),fs=100,channels=[f'e{i}' for i in range(8)],unit='uV')
            write_csv(root/'events.tsv',[{'onset':4,'duration':8,'trial_type':'seizure'}])
            (root/'events.tsv').write_text('onset\tduration\ttrial_type\n4\t8\tseizure\n')
            write_csv(root/'manifest.csv',[{'include':1,'path':'eeg.npz','subject':'chb21','group':'chb21','start_s':0,'stop_s':24,'events':'events.tsv','annotation_complete':1,'task':''}])
            cfg={'root':d,'manifest':str(root/'manifest.csv'),'cache':str(root/'cache'),'study':'chb','fs':128,'channels':[f'e{i}' for i in range(8)],'max_source_bytes':10**7,'max_cache_bytes':10**7,'seizure_column':'trial_type','seizure_values':['seizure']}
            prepare(cfg);x,y,g,ids=load_cache(cfg);self.assertEqual(len(x),6);self.assertEqual(y.sum(),2);self.assertEqual(set(g),{'chb01'})
            cp=root/'cache/record-00000.npz';cp.write_bytes(cp.read_bytes()+b'bad')
            with self.assertRaises(ValueError):load_cache(cfg)
    def test_features_at_both_sampling_rates(self):
        for fs in (100,128):
            x=self.rng.normal(size=(2,8,fs*4));z,q,a,s=features(x,fs)
            self.assertEqual(z.shape,(2,8,7));self.assertTrue(np.isfinite(z).all())

    def test_hbn_phenotype_join_and_task_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);names=[f'e{i}' for i in range(8)]
            np.savez(root/'eeg.npz',data=self.x.transpose(1,0,2).reshape(8,-1),fs=100,channels=names,unit='uV')
            write_csv(root/'manifest.csv',[{'include':1,'path':'eeg.npz','subject':'s1','group':'family1','start_s':0,'stop_s':24,'events':'','annotation_complete':0,'task':'rest'}])
            write_csv(root/'labels.csv',[{'subject':'s1','label':1,'group':'family1','label_source':'constructed unit-test label','task':'rest'}])
            cfg={'root':d,'manifest':str(root/'manifest.csv'),'labels':str(root/'labels.csv'),'cache':str(root/'cache'),'study':'hbn','fs':100,'channels':names,'max_source_bytes':10**7,'max_cache_bytes':10**7,'hbn_task':'rest','max_epochs_hbn':6}
            prepare(cfg);x,y,g,ids=load_cache(cfg);self.assertEqual(x[0].shape,(6,8,400));self.assertEqual(y.tolist(),[1]);self.assertEqual(g.tolist(),['family1'])
            cfg['cache']=str(root/'badcache');cfg['hbn_task']='different_task'
            with self.assertRaises(ValueError):prepare(cfg)

if __name__=='__main__':unittest.main()

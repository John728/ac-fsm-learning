"""Cache exact PYAC trajectories, then diagnose and repair terminal readout.

Run: python context_diagnostics.py capture --seed 41100
     python context_diagnostics.py analyze --seed 41100
No paper files or existing results are overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
import run_context_readout_ablation as old

OUT = ROOT / 'diagnostics' / 'context'
AREAS = ('state', 'transition', 'context')

def terminal(net):
    return np.array([net.assembly(a).indices for a in AREAS])

def capture(seed):
    OUT.mkdir(parents=True, exist_ok=True)
    net, symbols, words = old.build_matched_network(seed, 'full_sac')
    arrays = {'word_caps': np.array([words[w].indices for w in old.ALL_WORDS]),
              'initial': np.stack([net.weights[a, 'word'].toarray() for a in AREAS])}
    core_initial = {key: net.weights[key].copy() for key in old.CORE_KEYS}
    def frozen_word(w, rho=0, rng=None):
        net.clear_activity()
        states = []
        for ph in old.PRONUNCIATIONS[w]:
            asm = symbols[ph] if rho == 0 else old.corrupt_substitution(symbols[ph], rho, rng)
            # Evaluation is truly frozen, including normalization.
            net.step(clamps={'symbol': asm}, targets=('transition',), plasticity=False, normalize=False)
            net.clear_activity('symbol')
            net.step(targets=('state',), plasticity=False, normalize=False)
            net.step(targets=('context',), plasticity=False, normalize=False)
            states.append(terminal(net))
        return np.array(states)
    def eval_stage(stage):
        clean = [frozen_word(w) for w in old.ALL_WORDS]
        arrays[stage+'_clean'] = np.array([x[-1] for x in clean])
        arrays[stage+'_prefix'] = np.concatenate(clean)
        arrays[stage+'_native_weights'] = np.stack([net.weights[a, 'word'].toarray() for a in AREAS])
        for rho in (0.2, 0.4):
            rng = np.random.default_rng(seed + 9000 + round(100*rho))
            arrays[f'{stage}_noise{rho}'] = np.array([
                frozen_word(w, rho, rng)[-1] for rep in range(5) for w in old.ALL_WORDS])
    # A zero-core-plasticity control explicitly freezes its random core. Cache
    # its deterministic trajectories instead of recomputing unchanged steps.
    fixed={w:frozen_word(w)[-1] for w in old.ALL_WORDS} if old.BETA==0 else None
    def teach_fixed(w):
        net.clear_activity()
        for a,cap in zip(AREAS,fixed[w]):net.set_activity(a,cap)
        net.step(targets=('word',),forced_activations={'word':words[w]},plasticity=True)
    train, labels = [], []
    rng = np.random.default_rng(seed + 1000)
    order = list(old.BASE_WORDS)
    for ep in range(25):
        rng.shuffle(order)
        for w in order:
            if fixed is not None:teach_fixed(w)
            else:old.run_word(net, symbols, words, w, True, 'full_sac', teach=True)
            train.append(terminal(net)); labels.append(old.ALL_WORDS.index(w))
        if ep in (0, 4, 9, 14, 19, 24):
            arrays[f'epoch{ep+1}_clean'] = np.array([frozen_word(w)[-1] for w in old.BASE_WORDS])
            print(seed, 'base epoch', ep+1, flush=True)
    arrays['base_train'], arrays['base_labels'] = np.array(train), np.array(labels)
    eval_stage('base')
    train, labels = [], []
    rng = np.random.default_rng(seed + 2000)
    order = list(old.HELD_OUT_WORDS)
    for ep in range(20):
        rng.shuffle(order)
        for w in order:
            if fixed is not None:teach_fixed(w)
            else:
                net.clear_activity()
                plen = old.longest_base_prefix(w)
                for t, ph in enumerate(old.PRONUNCIATIONS[w]):
                    old.step_symbol(net, symbols[ph], t >= plen, 'full_sac')
                net.step(forced_activations={'word': words[w]}, targets=('word',), plasticity=True)
            train.append(terminal(net)); labels.append(old.ALL_WORDS.index(w))
    arrays['ext_train'], arrays['ext_labels'] = np.array(train), np.array(labels)
    eval_stage('post')
    # Fixed unlearned core control, retaining trained-core data above.
    for key, weights in core_initial.items():
        net.weights[key] = weights
    arrays['untrained_clean'] = np.array([frozen_word(w)[-1] for w in old.ALL_WORDS])
    np.savez_compressed(OUT / f'features_{seed}.npz', **arrays)
    manifest = {'seed': seed, 'words': old.ALL_WORDS, 'pronunciations': old.PRONUNCIATIONS,
                'base_epochs': 25, 'extension_epochs': 20, 'noise_trials': 5,
                'N': old.N, 'K': old.K, 'beta_core': old.BETA, 'beta_word': old.WORD_BETA,
                'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [Path(__file__), Path(old.__file__), ROOT/'data/cmudict.dict',
                   old.PYAC_SRC/'pyac/network.py', old.PYAC_SRC/'pyac/plasticity.py']}}
    (OUT / f'manifest_{seed}.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(seed, 'capture complete', flush=True)

def features(caps, areas):
    result = np.zeros((len(caps), len(areas)*old.N))
    for j, a in enumerate(areas):
        for i, cap in enumerate(caps[:, a]):
            result[i, j*old.N+cap] = 1
    return result

class Readout:
    """Exact dense implementation of fixed-k target Hebbian W updates.

    Only forced-target columns change; normalizing those columns is equivalent
    to full incoming L1 normalization, up to floating-point rounding.
    """
    def __init__(self, initial, word_caps, areas, beta=0.4):
        self.areas, self.caps, self.beta = areas, word_caps, beta
        self.w = np.concatenate([initial[a] for a in areas]).copy()
        self.w /= self.w.sum(axis=0, keepdims=True)
    def active(self, caps):
        return np.concatenate([caps[a]+i*old.N for i,a in enumerate(self.areas)])
    def fit(self, caps, labels, repeat=1):
        # Disjoint target assemblies and no W feedback make the final weights
        # depend only on source activation counts for each supervised label.
        # Column normalization commutes with later multiplicative updates.
        counts=np.zeros((self.w.shape[0],len(self.caps)),dtype=np.int32)
        for x,y in zip(caps,labels):
            counts[self.active(x),y]+=repeat
        for y in np.unique(labels):
            cols=self.caps[y]
            block=self.w[:,cols]*(1+self.beta)**counts[:,y,None]
            self.w[:, cols] = block / block.sum(axis=0, keepdims=True)
    def predict(self, caps, count):
        scores, drives = [], []
        for x in caps:
            drive = self.w[self.active(x)].sum(axis=0)
            winners = np.lexsort((np.arange(old.N), -drive))[:old.K]
            scores.append([np.isin(cap, winners).sum() for cap in self.caps[:count]])
            drives.append([drive[cap].mean() for cap in self.caps[:count]])
        return np.array(scores), np.array(drives)

def metrics(scores, labels):
    pred = scores.argmax(axis=1)
    ties = (scores == scores.max(axis=1, keepdims=True)).sum(axis=1)>1
    return {'accuracy': float(np.mean(pred==labels)),
            'strict_accuracy': float(np.mean((pred==labels)&~ties)),
            'ties': int(ties.sum()), 'errors': [[int(y),int(p)] for y,p in zip(labels,pred) if y!=p]}

def evaluate(r, d, stage):
    result = {}
    for key in ('clean','noise0.2','noise0.4'):
        x = d[stage+'_'+key]
        labels = np.tile(np.arange(36),len(x)//36)
        scores, drives = r.predict(x, 36)
        for subset, mask, count in (
            ('base28', labels<28, 28), ('base36',labels<28,36),
            ('extension',labels>=28,36), ('all',labels>=0,36)):
            if stage=='base' and subset in ('extension','all'): continue
            result[key+'_'+subset] = metrics(scores[mask,:count],labels[mask])
        result[key+'_drive_base28'] = metrics(drives[labels<28,:28], labels[labels<28])
    return result

def analyze(seed):
    d = dict(np.load(OUT/f'features_{seed}.npz'))
    result = {'geometry':{}, 'readouts':{}, 'validation':{}}
    result['epoch_geometry']={}
    for ep in (1,5,10,15,20,25):
        xx=features(d[f'epoch{ep}_clean'],(0,1))
        final=features(d['base_clean'][:28],(0,1))
        result['epoch_geometry'][str(ep)]={
            'distinct_terminal_sa':int(len(np.unique(xx,axis=0))),
            'mean_overlap_final':float(np.mean(np.sum(xx*final,axis=1)/60))}
    result['prefix_geometry']={}
    for stage,n in [('base',28),('post',36)]:
        prefixes={}; i=0
        for w in old.ALL_WORDS:
            seq=old.PRONUNCIATIONS[w]
            for t in range(len(seq)):
                if old.ALL_WORDS.index(w)<n:prefixes[seq[:t+1]]=d[stage+'_prefix'][i]
                i+=1
        for label,aa in [('S',(0,)),('SA',(0,1))]:
            groups={}
            for prefix,cap in prefixes.items():
                key=cap[list(aa)].tobytes()
                groups.setdefault(key,[]).append(' '.join(prefix))
            result['prefix_geometry'][stage+'_'+label]={
                'prefix_count':len(prefixes),'distinct_populations':len(groups),
                'collision_groups':[v for v in groups.values() if len(v)>1]}
    result['training_drift']={}
    for name,areas in [('S',(0,)),('A',(1,)),('C',(2,))]:
        train=features(d['base_train'],areas)
        target=features(d['base_clean'][d['base_labels']],areas)
        ovs=np.sum(train*target,axis=1)/old.K
        result['training_drift'][name]={
            'first5_epochs_to_final':float(np.mean(ovs[:140])),
            'last5_epochs_to_final':float(np.mean(ovs[-140:]))}
    for name, areas in [('S',(0,)),('A',(1,)),('SA',(0,1)),('C',(2,)),('SAC',(0,1,2))]:
        g = {}
        for stage in ('base','post','untrained'):
            x = features(d[stage+'_clean'],areas)
            n = 28 if stage in ('base','untrained') else 36
            x = x[:n]; gram = x@x.T
            groups = {}
            for w, xx in zip(old.ALL_WORDS[:n],x):
                groups.setdefault(xx.tobytes(),[]).append(w)
            groups = [v for v in groups.values() if len(v)>1]
            g[stage] = {'exact_collision_groups':groups,
                        'exact_pattern_ceiling': (n-sum(len(v)-1 for v in groups))/n,
                        'min_self_minus_competitor':float(np.min(np.diag(gram)-np.max(gram-np.eye(n)*1e9,axis=1)))}
            if stage=='untrained':continue
            # Ridge fit to one clean prototype per class is a diagnostic memory
            # readout; noise samples are never used to fit or select lambda.
            dual=np.linalg.solve(gram+np.eye(n),np.eye(n))
            # Canonical lookup is memorization/information access, not generalization.
            for key in ('clean','noise0.2','noise0.4'):
                xx=features(d[stage+'_'+key],areas)
                labels=np.tile(np.arange(36),len(xx)//36)
                mask=labels<n
                g[stage][key+'_nearest']=metrics(xx[mask]@x.T,labels[mask])
                g[stage][key+'_ridge']=metrics(xx[mask]@x.T@dual,labels[mask])
            if stage=='post':
                # Old templates are not refreshed after extension.
                templates=features(d['base_clean'][:28],areas)
                current=features(d['post_clean'][:28],areas)
                g[stage]['old_template_accuracy']=metrics(current@templates.T,np.arange(28))
                g[stage]['old_terminal_overlap']=float(np.mean(np.sum(current*templates,axis=1)/(old.K*len(areas))))
        result['geometry'][name]=g
    for name,areas in [('full',(0,1,2)),('sa',(0,1)),('s',(0,)),('c',(2,))]:
        r=Readout(d['initial'],d['word_caps'],areas)
        r.fit(d['base_train'],d['base_labels'])
        base=evaluate(r,d,'base')
        if name=='full':
            target=np.concatenate(list(d['base_native_weights']))
            result['validation']['base_native_max_abs_error']=float(np.max(np.abs(r.w-target)))
            native=Readout(d['initial'],d['word_caps'],areas)
            native.w=target.copy()
            result['validation']['base_native_prediction_disagreements']=int(np.sum(
                r.predict(d['base_clean'][:28],28)[0].argmax(axis=1)!=
                native.predict(d['base_clean'][:28],28)[0].argmax(axis=1)))
        r.fit(d['ext_train'],d['ext_labels'])
        result['readouts'][name+'_online']={'base':base,'post':evaluate(r,d,'post')}
        if name=='full':
            target=np.concatenate(list(d['post_native_weights']))
            result['validation']['post_native_max_abs_error']=float(np.max(np.abs(r.w-target)))
            native.w=target.copy()
            result['validation']['post_native_prediction_disagreements']=int(np.sum(
                r.predict(d['post_clean'],36)[0].argmax(axis=1)!=
                native.predict(d['post_clean'],36)[0].argmax(axis=1)))
            result['validation']['old_readout_weight_max_change']=float(np.max(np.abs(
                d['post_native_weights'][:,:,d['word_caps'][:28].ravel()]-
                d['base_native_weights'][:,:,d['word_caps'][:28].ravel()])))
        print(seed,name,'online',base['clean_base28']['accuracy'],flush=True)
    # Mechanistic interventions, each explicitly separated from original online protocol.
    for name, beta, repeat, frozen in [
        ('sa_repeat3',.4,3,False), ('sa_beta0.1',.1,1,False),
        ('sa_beta0.8',.8,1,False), ('sa_frozen',.4,1,True)]:
        r=Readout(d['initial'],d['word_caps'],(0,1),beta)
        if frozen:
            r.fit(d['base_clean'][d['base_labels']],d['base_labels'])
        else:r.fit(d['base_train'],d['base_labels'],repeat)
        base=evaluate(r,d,'base')
        r.fit(d['ext_train'],d['ext_labels'],repeat)
        result['readouts'][name]={'base':base,'post':evaluate(r,d,'post')}
        print(seed,name,base['clean_base28']['accuracy'],flush=True)
    # Capacity-matched control: the same C input graph but no recurrent
    # history. It only projects terminal S+A; exact input collisions persist.
    reference=old.Network(old.full_spec(),seed=seed)
    projection=np.concatenate([reference.weights[a,'context'].toarray() for a in AREAS[:2]])
    projection/=projection.sum(axis=0,keepdims=True)
    del reference
    projected=dict(d)
    memo={}
    for key in ['base_train','ext_train']+[s+'_'+k for s in ('base','post') for k in ('clean','noise0.2','noise0.4')]:
        values=d[key].copy()
        for cap in values:
            signature=cap[:2].tobytes()
            if signature not in memo:
                drive=projection[np.concatenate((cap[0],cap[1]+old.N))].sum(axis=0)
                memo[signature]=np.sort(np.lexsort((np.arange(old.N),-drive))[:old.K])
            cap[2]=memo[signature]
        projected[key]=values
    r=Readout(d['initial'],d['word_caps'],(0,1,2))
    r.fit(projected['base_train'],d['base_labels'])
    base=evaluate(r,projected,'base')
    r.fit(projected['ext_train'],d['ext_labels'])
    result['readouts']['sa_static_projection']={'base':base,'post':evaluate(r,projected,'post')}
    print(seed,'static projection',base['clean_base28']['accuracy'],flush=True)
    (OUT/f'analysis_{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(seed,'analysis complete',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['capture','analyze']);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--beta-core',type=float,default=.03)
    args=ap.parse_args()
    if args.beta_core != .03:
        old.BETA=args.beta_core
        OUT=OUT/f'core_beta_{args.beta_core}'
    globals()[args.mode](args.seed)

"""Physical four-area validation and exposure stress test of selected beta."""
import json
import numpy as np
import context_diagnostics as d

old=d.old
old.BETA=.003
seed=41100
net,symbols,words=old.build_matched_network(seed,'reduced_sa')

def recall(word):
    net.clear_activity()
    for ph in old.PRONUNCIATIONS[word]:
        net.step(clamps={'symbol':symbols[ph]},targets=('transition',),plasticity=False,normalize=False)
        net.clear_activity('symbol')
        net.step(targets=('state',),plasticity=False,normalize=False)
    pair=np.array([net.assembly(a).indices for a in ('state','transition')])
    net.step(targets=('word',),plasticity=False,normalize=False)
    return pair,net.assembly('word')

result={'seed':seed,'beta_core':.003,'areas':list(net.areas),'epochs':{}}
rng=np.random.default_rng(seed+1000);order=list(old.BASE_WORDS)
for ep in range(1,101):
    rng.shuffle(order)
    for w in order:old.run_word(net,symbols,words,w,True,'reduced_sa',teach=True)
    if ep in (1,5,10,25,50,75,100):
        pairs=[];hits=[]
        for w in old.BASE_WORDS:
            pair,out=recall(w);pairs.append(pair)
            scores={x:old.overlap(out,words[x]) for x in old.BASE_WORDS}
            hits.append(max(scores,key=scores.get)==w)
        result['epochs'][str(ep)]={'distinct_sa':len({p.tobytes() for p in pairs}),
                                  'native_accuracy':float(np.mean(hits))}
        print(ep,result['epochs'][str(ep)],flush=True)
        if ep==25:
            snap=net.snapshot()
            cache=np.load(d.OUT/'core_beta_0.003'/'features_41100.npz')
            result['physical_core_matches_cached_full_model']=bool(np.array_equal(np.array(pairs),cache['base_clean'][:28,:2]))
net.restore(snap)
rng=np.random.default_rng(seed+2000);order=list(old.HELD_OUT_WORDS)
for ep in range(20):
    rng.shuffle(order)
    for w in order:
        net.clear_activity()
        plen=old.longest_base_prefix(w)
        for t,ph in enumerate(old.PRONUNCIATIONS[w]):old.step_symbol(net,symbols[ph],t>=plen,'reduced_sa')
        net.step(targets=('word',),forced_activations={'word':words[w]},plasticity=True)
pairs=[];hits=[]
for w in old.ALL_WORDS:
    pair,out=recall(w);pairs.append(pair)
    scores={x:old.overlap(out,words[x]) for x in old.ALL_WORDS}
    hits.append(max(scores,key=scores.get)==w)
result['extension']={'base_accuracy':float(np.mean(hits[:28])), 'new_accuracy':float(np.mean(hits[28:])),
                     'core_matches_cached_full_model':bool(np.array_equal(np.array(pairs),cache['post_clean'][:,:2]))}
(d.OUT/'physical_four_area_and_exposure.json').write_text(json.dumps(result,indent=2)+'\n')
print(result,flush=True)

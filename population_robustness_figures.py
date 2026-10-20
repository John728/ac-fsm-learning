"""Population-level robustness, separation and preservation; measured four-area AC."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import run_context_readout_ablation as m
OUT=ROOT/'results'/'population_robustness_2026_09_22'
SEEDS=(41100,41200,41300,41400,41500)
STAGES=('untrained','base','extended')
TRIALS=10; RHO=.2
BLUE='#215F8B'; ORANGE='#CB6B35'; INK='#233746'; GREY='#A2ADB6'
PREFIXES=list(dict.fromkeys(seq[:i+1] for w in m.ALL_WORDS for seq in [m.PRONUNCIATIONS[w]] for i in range(len(seq))))
BASE=set(seq[:i+1] for w in m.BASE_WORDS for seq in [m.PRONUNCIATIONS[w]] for i in range(len(seq)))
OLD=np.array([p in BASE for p in PREFIXES]);NEW=~OLD

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def capture(seed):
    OUT.mkdir(parents=True,exist_ok=True);m.BETA=.003
    net,symbols,words=m.build_matched_network(seed,'reduced_sa')
    arrays={};proof={}
    def replay(seq,rng=None):
        net.clear_activity()
        for ph in seq:
            asm=symbols[ph] if rng is None else m.corrupt_substitution(symbols[ph],RHO,rng)
            net.step(clamps={'symbol':asm},targets=('transition',),plasticity=False,normalize=False)
            net.clear_activity('symbol')
            net.step(targets=('state',),plasticity=False,normalize=False)
        return net.assembly('state').indices.copy()
    def checkpoint(stage):
        before={str(k):hashlib.sha256(w.data.tobytes()).hexdigest() for k,w in net.weights.items()}
        arrays[stage+'_clean']=np.array([replay(p) for p in PREFIXES])
        # Identical corrupted input draws across checkpoints for each seed.
        rng=np.random.default_rng(seed+88000)
        arrays[stage+'_noisy']=np.array([[replay(p,rng) for p in PREFIXES] for _ in range(TRIALS)])
        after={str(k):hashlib.sha256(w.data.tobytes()).hexdigest() for k,w in net.weights.items()}
        assert before==after
        proof[stage+'_weights_unchanged_by_probes']=True
        print(seed,stage,'captured',flush=True)
    checkpoint('untrained')
    rng=np.random.default_rng(seed+1000);order=list(m.BASE_WORDS)
    for epoch in range(25):
        rng.shuffle(order)
        for w in order:m.run_word(net,symbols,words,w,True,'reduced_sa',teach=True)
        if (epoch+1)%5==0:print(seed,'base',epoch+1,flush=True)
    checkpoint('base')
    rng=np.random.default_rng(seed+2000);order=list(m.HELD_OUT_WORDS)
    for epoch in range(20):
        rng.shuffle(order)
        for w in order:
            net.clear_activity();plen=m.longest_base_prefix(w)
            for t,ph in enumerate(m.PRONUNCIATIONS[w]):m.step_symbol(net,symbols[ph],t>=plen,'reduced_sa')
            net.step(targets=('word',),forced_activations={'word':words[w]},plasticity=True)
    checkpoint('extended')
    np.savez_compressed(OUT/f'populations_{seed}.npz',**arrays)
    meta={'seed':seed,'areas':list(net.areas),'N':m.N,'k':m.K,'beta':m.BETA,'beta_W':m.WORD_BETA,
          'base_epochs':25,'extension_epochs':20,'rho':RHO,'noise_trials':TRIALS,
          'noise':'replace exactly 6 of 30 input neurons at every phoneme; fixed-k substitution',
          'prefixes':PREFIXES,'old_prefix_mask':OLD.tolist(),'base_words':m.BASE_WORDS,'extension_words':m.HELD_OUT_WORDS,
          'probe_validation':proof,'sha256':{str(p):sha(p) for p in [Path(__file__),Path(m.__file__),
              ROOT/'data/cmudict.dict',m.PYAC_SRC/'pyac/network.py',m.PYAC_SRC/'pyac/plasticity.py']}}
    (OUT/f'manifest_{seed}.json').write_text(json.dumps(meta,indent=2)+'\n')

def binary(caps):
    x=np.zeros((len(caps),m.N),dtype=np.float32)
    x[np.arange(len(caps))[:,None],caps]=1
    return x

def compare(noisy,canonical):
    # Each noisy population compared with every canonical prefix population.
    gram=(binary(noisy.reshape(-1,m.K))@binary(canonical).T).reshape(TRIALS,len(PREFIXES),len(PREFIXES))
    own=np.diagonal(gram,axis1=1,axis2=2).copy()
    for i in range(len(PREFIXES)):gram[:,i,i]=-1
    return own,gram.max(axis=2)

def analyze():
    result={'seeds':SEEDS,'prefix_counts':{'old':int(OLD.sum()),'new':int(NEW.sum())},'runs':{}}
    for seed in SEEDS:
        d=dict(np.load(OUT/f'populations_{seed}.npz'));r={}
        for stage in STAGES:
            clean=d[stage+'_clean'];noisy=d[stage+'_noisy'];own,rival=compare(noisy,clean)
            oldcanon=binary(clean[OLD]);oldnoisy=binary(noisy[:,OLD].reshape(-1,m.K))
            g=(oldnoisy@oldcanon.T).reshape(TRIALS,OLD.sum(),OLD.sum())
            for i in range(OLD.sum()):g[:,i,i]=-1
            oldrival=g.max(axis=2)
            r[stage]={'old_own':float(own[:,OLD].mean()),'old_rival':float(oldrival.mean()),
                      'old_positive_margin_fraction':float(np.mean(own[:,OLD]>oldrival)),
                      'old_distinct_clean':len(np.unique(clean[OLD],axis=0)),
                      'new_own':float(own[:,NEW].mean()),'new_rival_all_histories':float(rival[:,NEW].mean()),
                      'all_distinct_clean':len(np.unique(clean,axis=0)),
                      'old_own_counts':own[:,OLD].tolist(),'new_own_counts':own[:,NEW].tolist()}
        pre=d['base_clean'];post=d['extended_clean']
        retained=np.array([len(set(a)&set(b)) for a,b in zip(pre,post)])
        oldreference,_=compare(d['extended_noisy'],pre)
        r['preservation']={'old_clean_retained':float(retained[OLD].mean()),
                           'old_clean_retained_by_prefix':retained[OLD].tolist(),
                           'old_exact_retained_count':int(np.sum(retained[OLD]==30)),
                           'old_noisy_pre_extension_reference':float(oldreference[:,OLD].mean()),
                           'new_clean_retained':float(retained[NEW].mean())}
        for x in d.values():assert np.all(np.diff(x,axis=-1)>0)
        # Compare all clean prefixes to the independent context diagnostic cache.
        cache=np.load(ROOT/f'diagnostics/context/core_beta_0.003/features_{seed}.npz')
        for stage,key in [('base','base_prefix'),('extended','post_prefix')]:
            offsets={};i=0
            for w in m.ALL_WORDS:
                seq=m.PRONUNCIATIONS[w]
                for t in range(len(seq)):offsets[seq[:t+1]]=cache[key][i,0];i+=1
            assert all(np.array_equal(d[stage+'_clean'][j],offsets[p]) for j,p in enumerate(PREFIXES))
        r['clean_prefixes_match_independent_capture']=True
        result['runs'][str(seed)]=r
    result['render_source_sha256']=sha(__file__)
    (OUT/'measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def save(fig,name):
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=250,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def style(ax):
    ax.spines[['top','right']].set_visible(False);ax.tick_params(length=3,color=GREY)
    ax.grid(axis='y',color='#E6EBEF',lw=.7);ax.set_axisbelow(True)

def seedlines(ax,values,x,color,label):
    a=np.array(values)
    for row in a:ax.plot(x,row,color=color,alpha=.24,lw=.9,marker='o',ms=3)
    ax.plot(x,a.mean(axis=0),color=color,lw=2.5,marker='o',ms=6,label=label)

def population_maps():
    data=dict(np.load(OUT/'populations_41100.npz'))
    fig,axes=plt.subplots(2,3,figsize=(11,7.5))
    fig.subplots_adjust(left=.07,right=.88,bottom=.16,top=.82,wspace=.15,hspace=.22)
    fig.suptitle('Do noisy inputs recruit the same neurons?',x=.07,ha='left',fontsize=18,fontweight='bold')
    fig.text(.07,.905,'Same 10 corrupted inputs before and after learning · 20% of active input neurons replaced at every phoneme',fontsize=10)
    c=plt.get_cmap('Blues');coords=np.arange(m.N);x=coords%50;y=coords//50
    evidence={}
    for col,w in enumerate(['camp','word','dock']):
        i=PREFIXES.index(m.PRONUNCIATIONS[w]);evidence[w]={}
        for row,stage in enumerate(['untrained','base']):
            ax=axes[row,col];caps=data[stage+'_noisy'][:,i];reference=data[stage+'_clean'][i]
            counts=np.bincount(caps.ravel(),minlength=m.N)/TRIALS
            active=np.flatnonzero(counts);own=np.mean([len(set(cap)&set(reference)) for cap in caps])
            ax.scatter(x[active],y[active],c=counts[active],cmap=c,vmin=0,vmax=1,s=12,edgecolors='none')
            ax.scatter(x[reference],y[reference],s=20,facecolors='none',edgecolors=ORANGE,linewidths=.65)
            ax.set(xlim=(-2,51),ylim=(41,-2),aspect='equal');ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if row==0:ax.set_title(f'{w}  /'+ ' '.join(m.PRONUNCIATIONS[w])+'/',fontsize=11,fontweight='bold',pad=9)
            ax.text(.5,-.07,f'{own:.1f} / 30 clean-reference neurons retained',transform=ax.transAxes,
                    ha='center',fontsize=9,color=BLUE)
            if col==0:ax.set_ylabel('Before learning' if row==0 else 'After base learning',fontsize=12,labelpad=13,fontweight='bold')
            evidence[w][stage]={'mean_reference_neurons_retained':own,'clean_neurons':reference.tolist(),
                                'activation_frequency':counts.tolist()}
    cax=fig.add_axes([.91,.28,.018,.42]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap=c),cax=cax)
    cb.set_ticks([0,.5,1],labels=['0 / 10','5 / 10','10 / 10']);cb.set_label('Noisy trials activating this neuron',labelpad=10)
    cb.outline.set_visible(False)
    fig.legend(handles=[Line2D([0],[0],marker='o',markerfacecolor='none',markeredgecolor=ORANGE,ls='',
                               label='Orange ring: active in the clean replay at this checkpoint')],
               loc='lower left',bbox_to_anchor=(.065,.075),frameon=False,fontsize=10)
    fig.text(.07,.04,'Each location is one of 2,000 S neurons. Positions are fixed across all panels; blank means inactive in all noisy trials.',fontsize=9)
    fig.text(.07,.014,'Representative seed 41100; words chosen in advance. Grid positions are neuron IDs, not anatomical coordinates.',fontsize=9)
    save(fig,'01_neurons_under_corruption')
    return evidence

def robustness(result):
    runs=list(result['runs'].values());fig,axes=plt.subplots(1,2,figsize=(11,5.5))
    fig.subplots_adjust(left=.08,right=.97,top=.79,bottom=.22,wspace=.28)
    fig.suptitle('Robustness must preserve the distinction between histories',x=.08,ha='left',fontsize=17,fontweight='bold')
    fig.text(.08,.895,'51 base histories · 10 corrupted trials per history · 5 matched seeds · each seed contributes equally',fontsize=10)
    labels=['Untrained','Base learned','After extension'];x=np.arange(3)
    ax=axes[0];style(ax)
    for key,color,label in [('old_own',BLUE,'Matching clean history'),('old_rival',ORANGE,'Closest other clean history')]:
        seedlines(ax,[[r[s][key] for s in STAGES] for r in runs],x,color,label)
    ax.set(ylim=(0,30),xticks=x,xticklabels=labels,ylabel='Shared S neurons (out of 30)')
    ax.set_title('A   Noise consistency versus competing histories',loc='left',fontsize=11,fontweight='bold')
    ax.legend(frameon=False,fontsize=9,loc='upper left')
    ax=axes[1];style(ax)
    vals=np.array([[r[s]['old_distinct_clean'] for s in STAGES] for r in runs])
    # Fixed horizontal seed offsets reveal separate replicates, no value jitter.
    for i,row in enumerate(vals):ax.plot(x+(i-2)*.035,row,'o-',color=BLUE,alpha=.55,lw=1,ms=4)
    ax.axhline(51,color=GREY,lw=1,ls='--')
    ax.set(ylim=(0,54),xticks=x,xticklabels=labels,ylabel='Distinct clean S populations (out of 51)')
    ax.set_title('B   Check that consistency is not collapse',loc='left',fontsize=11,fontweight='bold')
    for j,stage in enumerate(STAGES):
        ax.text(j,vals[:,j].min()-5,f'{vals[:,j].min()}–{vals[:,j].max()}' if vals[:,j].min()!=vals[:,j].max() else str(vals[0,j]),
                ha='center',fontsize=10,color=BLUE)
    fig.text(.08,.09,'Thin lines / dots: seed means. Thick lines: mean across seeds. References in panel A are from the same checkpoint.',fontsize=9)
    fig.text(.08,.045,'“Other history” is the strongest-overlapping competitor among the same 51 base histories. Exact equality defines a collision.',fontsize=9)
    save(fig,'02_robustness_and_separation')

def preservation(result):
    runs=list(result['runs'].values());fig,axes=plt.subplots(1,3,figsize=(12,5.6))
    fig.subplots_adjust(left=.075,right=.98,top=.79,bottom=.24,wspace=.4)
    fig.suptitle('What survives when new sequences are learned?',x=.075,ha='left',fontsize=18,fontweight='bold')
    fig.text(.075,.895,'Old histories and new suffix histories measured separately · 5 seeds · 20 extension epochs without base-word replay',fontsize=10)
    specs=[('A   Old populations: clean replay',[[30,r['preservation']['old_clean_retained']] for r in runs]),
           ('B   Old populations: corrupted replay',[[r['base']['old_own'],r['preservation']['old_noisy_pre_extension_reference']] for r in runs]),
           ('C   New suffixes: corrupted replay',[[r['base']['new_own'],r['extended']['new_own']] for r in runs])]
    for j,(ax,(title,values)) in enumerate(zip(axes,specs)):
        style(ax);seedlines(ax,values,[0,1],BLUE,'Matching history')
        if j==2:seedlines(ax,[[r['base']['new_rival_all_histories'],r['extended']['new_rival_all_histories']] for r in runs],
                         [0,1],ORANGE,'Closest other history')
        ax.set(ylim=(0,31),xlim=(-.17,1.17),xticks=[0,1],xticklabels=['Before\nextension','After\nextension'])
        ax.set_title(title,loc='left',fontsize=10.5,fontweight='bold')
        means=np.mean(values,axis=0)
        for x,y in enumerate(means):ax.annotate(f'{y:.1f}/30',(x,y),xytext=(0,-16),textcoords='offset points',ha='center',fontsize=10,color=BLUE)
        ax.text(.5,-.26,'51 old histories\nFixed pre-extension reference' if j<2 else '12 new suffix histories\nReference from each checkpoint',
                transform=ax.transAxes,ha='center',fontsize=9,color=INK)
    axes[0].set_ylabel('Shared S neurons (out of 30)')
    axes[2].legend(frameon=False,fontsize=8,loc='upper left')
    fig.text(.075,.055,'Panels A–B test preservation of the original populations; panel C tests noise consistency, not identity preservation.',fontsize=9)
    fig.text(.075,.015,'Thin lines / dots: seed means. Thick lines: mean across seeds. All corrupted probes use 20% neuron substitution at every input.',fontsize=9)
    save(fig,'03_population_preservation')

def render():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':INK,'axes.labelcolor':INK,
                         'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'svg.fonttype':'none'})
    result=analyze();evidence=population_maps();robustness(result);preservation(result)
    (OUT/'illustrative_neurons.json').write_text(json.dumps(evidence,indent=2)+'\n')
    for stage in STAGES:
        print(stage,{key:float(np.mean([r[stage][key] for r in result['runs'].values()])) for key in
                     ['old_own','old_rival','old_distinct_clean','new_own','new_rival_all_histories']})
    print('preservation',{key:float(np.mean([r['preservation'][key] for r in result['runs'].values()])) for key in
                          ['old_clean_retained','old_exact_retained_count','old_noisy_pre_extension_reference']})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['capture','render']);ap.add_argument('--seed',type=int,default=41100)
    args=ap.parse_args()
    capture(args.seed) if args.mode=='capture' else render()

"""One held-order population-raster prototype on the existing length-20 task."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import run_4area_reuse_sweep as task
OUT=ROOT/'results'/'sequence_raster_prototype_2026_09_22'
SEEDS=(101,202,303,404,505);TRIALS=10;RHO=.2
INK='#253847';BLUE='#23628B';ORANGE='#C36636'

def capture(seed):
    OUT.mkdir(parents=True,exist_ok=True)
    seqs,_=task.generate_dataset(42)
    labels=list(seqs)+[f'ext_r{r}' for r in task.R_VALUES]
    net,symbols,words=task.build_network(seed,task.ALPHABET,labels)
    def replay(seq,rng=None):
        net.clear_activity();states=[]
        for ph in seq:
            cap=symbols[ph]
            if rng is not None:
                keep=rng.choice(cap.indices,24,replace=False)
                other=rng.choice(np.setdiff1d(np.arange(task.N),cap.indices),6,replace=False)
                cap=task.Assembly('symbol',np.sort(np.r_[keep,other]))
            net.step(clamps={'symbol':cap},targets=('transition',),plasticity=False,normalize=False)
            net.clear_activity('symbol')
            net.step(targets=('state',),plasticity=False,normalize=False)
            states.append(net.assembly('state').indices.copy())
        return np.array(states)
    arrays={};validation={}
    def checkpoint(stage):
        before=[hashlib.sha256(w.data.tobytes()).hexdigest() for w in net.weights.values()]
        arrays[stage+'_clean']=np.array([replay(seq) for seq in seqs.values()])
        rng=np.random.default_rng(seed+90000)
        arrays[stage+'_noise']=np.array([[replay(seq,rng) for seq in seqs.values()] for _ in range(TRIALS)])
        assert before==[hashlib.sha256(w.data.tobytes()).hexdigest() for w in net.weights.values()]
        validation[stage+'_weights_unchanged']=True
        print(seed,stage,'complete',flush=True)
    checkpoint('before')
    for _ in range(task.BASE_EPOCHS):
        for label,seq in seqs.items():task.run_sequence(net,symbols,words,seq,label,True,True)
    checkpoint('after')
    np.savez_compressed(OUT/f'capture_{seed}.npz',**arrays)
    meta={'seed':seed,'dataset_seed':42,'sequences':seqs,'areas':list(net.areas),'N':task.N,'k':task.K,
          'core_beta':task.BETA,'word_beta':task.WORD_BETA,'epochs':task.BASE_EPOCHS,
          'trials':TRIALS,'rho':RHO,'noise_rng_seed':seed+90000,'validation':validation,
          'sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                    [__file__,task.__file__,task.PYAC_SRC/'pyac/network.py',task.PYAC_SRC/'pyac/plasticity.py']}}
    (OUT/f'manifest_{seed}.json').write_text(json.dumps(meta,indent=2)+'\n')

def activity(caps):
    out=np.zeros((task.N,task.L),dtype=float)
    for t,cap in enumerate(caps):out[cap,t]=1
    return out

def render(dataset='long'):
    global OUT,SEEDS
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':INK,'axes.labelcolor':INK,
                         'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'svg.fonttype':'none'})
    lexical=dataset=='phoneme'
    if lexical:
        import population_robustness_figures as p
        SEEDS=p.SEEDS; selected=['camp','word','dock']
        prefix_ids=[p.PREFIXES.index(p.m.PRONUNCIATIONS[w][:t+1]) for w in selected for t in range(len(p.m.PRONUNCIATIONS[w]))]
        task.L=len(prefix_ids);task.B=1
        runs={}
        for seed in SEEDS:
            source=dict(np.load(p.OUT/f'populations_{seed}.npz'))
            runs[seed]={}
            for stage,original in [('before','untrained'),('after','base')]:
                runs[seed][stage+'_clean']=source[original+'_clean'][prefix_ids][None]
                runs[seed][stage+'_noise']=source[original+'_noisy'][:,prefix_ids][:,None]
        OUT=OUT/'phoneme';OUT.mkdir(parents=True,exist_ok=True)
    else:runs={seed:dict(np.load(OUT/f'capture_{seed}.npz')) for seed in SEEDS}
    d=runs[SEEDS[0]];reference=d['after_clean'][0]
    # Every neuron active anywhere in this clean trajectory appears once.
    # Sort by first clean activation; ties break by neuron ID. Never use noise.
    order=sorted(set(reference.ravel()),key=lambda n:(next(t for t,c in enumerate(reference) if n in c),int(n)))
    order=np.array(order,dtype=int)
    other=np.setdiff1d(np.arange(task.N),order)
    measures={'illustrative_seed':SEEDS[0],'illustrative_sequence':selected if lexical else 'base_0','neuron_order':order.tolist(),
              'dataset':dataset,'ordering':'first activation in displayed trained clean reference, neuron ID tie break; never fitted to noise',
              'runs':{}}
    if lexical:
        measures['source_hashes']={str(p.OUT/f'populations_{s}.npz'):hashlib.sha256((p.OUT/f'populations_{s}.npz').read_bytes()).hexdigest() for s in SEEDS}
        measures['prefix_ids']=prefix_ids
    for seed,run in runs.items():
        out={}
        for stage in ['before','after']:
            clean=run[stage+'_clean'];noise=run[stage+'_noise']
            counts=np.array([[[len(set(noise[r,w,t])&set(clean[w,t])) for t in range(task.L)]
                               for w in range(task.B)] for r in range(TRIALS)])
            out[stage]={'mean_correct_members':float(counts.mean()),'by_step':counts.mean(axis=(0,1)).tolist()}
        measures['runs'][str(seed)]=out
    # Two row blocks; each raster has an explicit off-reference count strip.
    fig=plt.figure(figsize=(12.4,8.8))
    grid=fig.add_gridspec(2,3,left=.075,right=.975,top=.80,bottom=.21,width_ratios=[1,1,.68],wspace=.23,hspace=.44)
    fig.suptitle('Learning stabilises neuronal sequences under noise' if lexical else 'A neural sequence under corrupted input',x=.075,ha='left',fontsize=20,fontweight='bold')
    fig.text(.075,.91,('Three words, separately reset · fixed neuron order · same corrupted inputs before and after learning' if lexical else
                      'Length-20 sequence · one fixed neuron order · no smoothing or interpolation'),fontsize=10.5)
    fields={}
    for row,stage in enumerate(['before','after']):
        clean=activity(d[stage+'_clean'][0])
        noisy=np.mean([activity(cap) for cap in d[stage+'_noise'][:,0]],axis=0)
        fields[stage]={}
        for col,(name,mat) in enumerate([('Clean replay',clean),('Corrupted replays',noisy)]):
            inner=grid[row,col].subgridspec(2,1,height_ratios=[7,1],hspace=.10)
            ax=fig.add_subplot(inner[0]);strip=fig.add_subplot(inner[1],sharex=ax)
            ax.imshow(mat[order],aspect='auto',interpolation='nearest',cmap='Greys',vmin=0,vmax=1,
                      extent=(.5,task.L+.5,len(order)+.5,.5),rasterized=True)
            ax.set_xticks([]);ax.set_yticks([1,len(order)]);ax.tick_params(length=0)
            for spine in ax.spines.values():spine.set_visible(False)
            if row==0:ax.set_title(name,loc='left',fontsize=13,fontweight='bold',pad=30 if lexical else 12)
            if col==0:ax.set_ylabel(('Before learning' if row==0 else ('After 25 epochs' if lexical else 'After 6 epochs'))+'\n\nReference-neuron row',fontsize=11,labelpad=9)
            # All excluded neurons accounted for, not silently hidden.
            count=mat[other].sum(axis=0)
            strip.bar(np.arange(1,task.L+1),count,width=.85,color=ORANGE,linewidth=0)
            strip.set(ylim=(0,30),yticks=[0,30],xticks=[1,4,7,10] if lexical else [1,5,10,15,20]);strip.tick_params(labelsize=8,length=2)
            ax.tick_params(axis='x',labelbottom=False,bottom=False)
            strip.spines[['top','right']].set_visible(False)
            if row==1:
                if lexical:
                    phonemes=[ph for w in selected for ph in p.m.PRONUNCIATIONS[w]]
                    strip.set_xticks(np.arange(1,task.L+1),phonemes)
                    strip.set_xlabel('Presented phoneme',fontsize=10)
                else:strip.set_xlabel('Input step',fontsize=10)
            if col==0:strip.set_ylabel('Other\nneurons',fontsize=8,rotation=0,labelpad=22,va='center')
            if lexical:
                for pos,word in [(2.5,'camp'),(6,'word'),(9,'dock')]:
                    ax.text((pos-.5)/task.L,1.025,word,transform=ax.transAxes,ha='center',fontsize=9,color=INK)
                for pos in [4.5,7.5]:
                    ax.axvline(pos,color='#B7C0C7',lw=.8,ls='--');strip.axvline(pos,color='#B7C0C7',lw=.8,ls='--')
            fields[stage][name]={'mean_off_reference_count':float(count.mean())}
    ax=fig.add_subplot(grid[:,2]);ax.spines[['top','right']].set_visible(False)
    for stage,color,label in [('before',ORANGE,'Before learning'),('after',BLUE,'After learning')]:
        a=np.array([measures['runs'][str(seed)][stage]['by_step'] for seed in SEEDS])
        for row in a:ax.plot(np.arange(1,task.L+1),row,color=color,alpha=.25,lw=.8)
        ax.plot(np.arange(1,task.L+1),a.mean(axis=0),color=color,lw=2.4,label=label)
    ax.set(ylim=(0,30),xlim=(1,task.L),xticks=[1,4,7,10] if lexical else [1,10,20],xlabel='Input step',ylabel='Matching clean-reference neurons (out of 30)')
    ax.set_title('Across five seeds',loc='left',fontsize=12,fontweight='bold',pad=30 if lexical else 12)
    ax.legend(frameon=False,fontsize=9,loc='center right' if lexical else 'upper right');ax.grid(axis='y',color='#E7EBEE',lw=.6)
    if lexical:
        for pos in [4.5,7.5]:ax.axvline(pos,color='#B7C0C7',lw=.8,ls='--')
        for pos,w in [(2.5,'camp'),(6,'word'),(9,'dock')]:ax.text((pos-1)/(task.L-1),1.02,w,transform=ax.transAxes,ha='center',fontsize=9)
    ax.text(.0,-.14,('Same three words · 10 probes per prefix' if lexical else '8 sequences × 10 noisy trials')+'\nThin lines = seed means',transform=ax.transAxes,fontsize=8)
    # Figure-level binary/frequency key with short, explicit scope.
    cax=fig.add_axes([.075,.102,.19,.013]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='Greys'),cax=cax,orientation='horizontal')
    cb.set_ticks([0,1],labels=['Never active','Always active']);cb.ax.tick_params(labelsize=8,length=0);cb.outline.set_visible(False)
    fig.text(.30,.10,f'Raster: {len(order)} neurons in the trained clean reference. Orange strips count all other neurons.',fontsize=9)
    fig.text(.075,.049,('Noise replaces 6/30 input neurons at every phoneme. Shading: mean of 10 independent noisy prefix probes; no smoothing.' if lexical else
                       'Noise replaces 6/30 input neurons at every step. Corrupted panels average the same 10 input draws across checkpoints.'),fontsize=9)
    fig.text(.075,.012,'The clean band is created by ordering; its persistence in independent corrupted inputs—not its shape—is the evidence.',fontsize=9)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'sequence_raster.{ext}',dpi=260,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    measures['display']=fields
    measures['render_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'measurements.json').write_text(json.dumps(measures,indent=2)+'\n')
    print('Displayed neurons',len(order))
    for stage in ['before','after']:
        values=[measures['runs'][str(s)][stage]['mean_correct_members'] for s in SEEDS]
        print(stage,np.mean(values),np.std(values,ddof=1),values)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['capture','render']);ap.add_argument('--seed',type=int,default=101)
    ap.add_argument('--dataset',choices=['long','phoneme'],default='long')
    args=ap.parse_args();capture(args.seed) if args.mode=='capture' else render(args.dataset)

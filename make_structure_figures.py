"""Measured four-area structure figures. Run capture, then render (no paper writes)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Wedge, Circle

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import run_context_readout_ablation as model
OUT=ROOT/'results'/'structure_figures_2026_09_22'
WORDS=('cat','can','camp','cats','candy','campus')
BLUE='#245A81'; ORANGE='#CB6B35'; INK='#213442'; GREY='#9BA7AE'

def capture():
    OUT.mkdir(parents=True,exist_ok=True)
    model.BETA=.003
    net,symbols,words=model.build_matched_network(41100,'reduced_sa')
    def probe():
        result={}
        for w in WORDS:
            net.clear_activity(); caps=[]
            for ph in model.PRONUNCIATIONS[w]:
                net.step(clamps={'symbol':symbols[ph]},targets=('transition',),plasticity=False,normalize=False)
                net.clear_activity('symbol')
                net.step(targets=('state',),plasticity=False,normalize=False)
                caps.append(net.assembly('state').indices.copy())
            result[w]=np.array(caps)
            net.step(targets=('word',),plasticity=False,normalize=False)
            output=net.assembly('word')
            result[w+'_output']=np.array([model.overlap(output,words[label]) for label in model.ALL_WORDS])
        return result
    probes=[probe()]
    rng=np.random.default_rng(42100); order=list(model.BASE_WORDS)
    for ep in range(25):
        rng.shuffle(order)
        for w in order:model.run_word(net,symbols,words,w,True,'reduced_sa',teach=True)
        probes.append(probe())
        print('base',ep+1,flush=True)
    rng=np.random.default_rng(43100); order=list(model.HELD_OUT_WORDS)
    for ep in range(20):
        rng.shuffle(order)
        for w in order:
            net.clear_activity(); plen=model.longest_base_prefix(w)
            for t,ph in enumerate(model.PRONUNCIATIONS[w]):
                model.step_symbol(net,symbols[ph],t>=plen,'reduced_sa')
            net.step(targets=('word',),forced_activations={'word':words[w]},plasticity=True)
        probes.append(probe())
        print('extension',ep+1,flush=True)
    data={w:np.stack([p[w] for p in probes]) for w in WORDS}
    for x in data.values():
        assert x.shape[0]==46 and x.shape[-1]==30
        assert all(len(np.unique(cap))==30 for cap in x.reshape(-1,30))
    data.update({w+'_output':np.stack([p[w+'_output'] for p in probes]) for w in WORDS})
    np.savez_compressed(OUT/'population_probes.npz',**data)
    metadata={'seed':41100,'areas':list(net.areas),'core_beta':.003,'word_beta':.4,
              'base_epochs':25,'extension_epochs':20,'probe_epochs':list(range(46)),
              'words':WORDS,'pronunciations':{w:model.PRONUNCIATIONS[w] for w in WORDS},
              'base_vocabulary':model.BASE_WORDS,'extension_vocabulary':model.HELD_OUT_WORDS,
              'probe':'frozen clean replay after each epoch; epoch 0 is untrained',
              'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [Path(__file__),Path(model.__file__),ROOT/'data/cmudict.dict',
                          model.PYAC_SRC/'pyac/network.py',model.PYAC_SRC/'pyac/plasticity.py']}}
    (OUT/'capture_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')

def save(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=240,facecolor='white',bbox_inches='tight')
    plt.close(fig)

def prefix_caps(data,epoch):
    result={}
    for w in WORDS:
        seq=model.PRONUNCIATIONS[w]
        for t,cap in enumerate(data[w][epoch]):
            p=seq[:t+1]
            if p in result:assert np.array_equal(result[p],cap)
            result[p]=cap
    return result

def graph(data):
    # Prefixes set layout only. Exact observed S membership sets node identity;
    # multiple histories with identical caps are collapsed to a single node.
    base=prefix_caps(data,25)
    layout={('K',):(0,0),('K','AE'):(1,0)}
    for w,y in [('cats',1.6),('candy',0),('campus',-1.6)]:
        seq=model.PRONUNCIATIONS[w]
        for t in range(2,len(seq)):layout[seq[:t+1]]=(t,y)
    trained={model.PRONUNCIATIONS[w][:t+1] for w in WORDS[:3]
             for t in range(len(model.PRONUNCIATIONS[w]))}
    ids={}; snapshots=[]
    fig,axes=plt.subplots(3,1,figsize=(11.4,9.2))
    fig.subplots_adjust(top=.90,bottom=.075,hspace=.40,left=.065,right=.91)
    fig.suptitle('Observed neural paths through lexical extension',x=.065,ha='left',fontsize=19,fontweight='bold')
    fig.text(.065,.935,'Six-word view of a 36-word network · fixed layout · each node is an observed 30-neuron S population',fontsize=10,color=INK)
    for ax,epoch,title in zip(axes,[25,26,45],['A   Before extension','B   After one extension epoch','C   After twenty extension epochs']):
        caps=prefix_caps(data,epoch); groups={}
        for p,c in caps.items():groups.setdefault(tuple(c),[]).append(p)
        positions={k:np.mean([layout[p] for p in pp],axis=0) for k,pp in groups.items()}
        for k in groups:
            if k not in ids:ids[k]=f'S{len(ids)+1:02d}'
        for p,c in caps.items():
            if len(p)==1:continue
            start=positions[tuple(caps[p[:-1]])]; end=positions[tuple(c)]
            novel=p not in trained; color=GREY if novel and epoch==25 else ORANGE if novel else BLUE
            ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'-|>','color':color,'lw':1.7,
                        'linestyle':'--' if novel and epoch==25 else '-','shrinkA':20,'shrinkB':20})
            middle=(start+end)/2
            ax.text(middle[0],middle[1]+.14,p[-1],ha='center',fontsize=9,color=INK,
                    bbox={'facecolor':'white','edgecolor':'none','pad':1})
        nodes=[]
        for k,pp in groups.items():
            x,y=positions[k]; novel=all(p not in trained for p in pp)
            color=GREY if novel and epoch==25 else ORANGE if novel else BLUE
            ax.scatter([x],[y],s=720,facecolor='white',edgecolor=color,linewidth=2,zorder=3)
            ax.text(x,y,ids[k],ha='center',va='center',fontsize=8,fontweight='bold',color=INK,zorder=4)
            counts=[len(set(k)&set(base[p])) for p in pp]
            if epoch>25 and any(n<30 for n in counts):
                ax.text(x,y-.70,('/'.join(map(str,counts)))+'/30',ha='center',fontsize=8,color=color)
            for w in WORDS:
                if model.PRONUNCIATIONS[w] in pp:
                    target=data[w+'_output'][epoch,model.ALL_WORDS.index(w)]
                    ax.text(x,y+.66,f'{w} · {target:.2f}',ha='center',fontsize=9,fontweight='bold',color=color)
            nodes.append({'id':ids[k],'prefixes':[list(p) for p in pp],'neurons':list(map(int,k)),
                          'overlap_counts_with_pre_extension':counts})
        snapshots.append({'epoch':epoch,'nodes':nodes})
        ax.text(-.42,0,'K →',ha='right',va='center',fontsize=9,color=INK)
        ax.set(xlim=(-.75,5.6),ylim=(-2.3,2.6)); ax.axis('off')
        ax.set_title(title,loc='left',fontsize=12,fontweight='bold',pad=2)
        if epoch>25:
            unchanged=sum(np.array_equal(c,base[p]) for p,c in caps.items())
            ax.text(1,1.03,f'{unchanged}/{len(caps)} populations unchanged (30/30 members)',
                    transform=ax.transAxes,ha='right',fontsize=9,color=INK)
    handles=[Line2D([0],[0],color=c,lw=2,linestyle=ls,label=lab) for c,ls,lab in
             [(BLUE,'-','Previously trained prefix'),(ORANGE,'-','Extension suffix'),(GREY,'--','Untrained suffix probe')]]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.06,.015),ncol=3,frameon=False,fontsize=9)
    fig.text(.065,.005,'Word labels report target W-population overlap (not a probability). Node IDs track exact S-neuron membership.',fontsize=9,color=INK)
    save(fig,'01_observed_machine')
    return snapshots

def raster(data):
    fig,axes=plt.subplots(2,1,figsize=(11.4,7.6),sharex=True)
    fig.subplots_adjust(top=.87,bottom=.14,hspace=.55,left=.10,right=.97)
    fig.suptitle('Who stays in the assembly?',x=.10,ha='left',fontsize=19,fontweight='bold')
    fig.text(.10,.92,'Frozen population probes after every training epoch · actual neuron membership, not similarity to an ideal state',fontsize=10,color=INK)
    stats={}
    for ax,w,title in zip(axes,['camp','campus'],['A   Established prefix: /K AE M P/','B   Extension endpoint: /K AE M P AH S/']):
        caps=data[w][:,-1,:]; ref=set(caps[25]); union=set(caps.ravel())
        # Fixed order: pre-extension members first, then other observed neurons
        # by first appearance. Numeric IDs break ties; row map is exported.
        first={int(n):next(i for i,c in enumerate(caps) if n in c) for n in union}
        neurons=sorted(union,key=lambda n:(n not in ref,first[int(n)],int(n)))
        mat=np.zeros((len(neurons),46),dtype=np.uint8)
        for i,n in enumerate(neurons):mat[i,np.any(caps==n,axis=1)]=1 if n in ref else 2
        ax.imshow(mat,aspect='auto',interpolation='nearest',cmap=ListedColormap(['#FAFAF8',BLUE,ORANGE]),
                  vmin=0,vmax=2,extent=(-.5,45.5,len(neurons)+.5,.5),rasterized=True)
        ax.axvline(25.5,color=INK,lw=1.3,ls='--')
        ax.axhline(30.5,color='#C9CED0',lw=.8)
        retained=len(set(caps[-1])&ref)
        ax.set_title(title,loc='left',fontweight='bold',fontsize=12,pad=12)
        ax.text(1,1.055,f'{retained}/30 members retained after extension',transform=ax.transAxes,
                ha='right',fontsize=9,color=INK)
        ax.set_ylabel('Neuron row\n(fixed ordering)',fontsize=10)
        ax.set_yticks([1,30,len(neurons)]); ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        stats[w]={'neuron_row_ids':list(map(int,neurons)),'unique_neurons':len(neurons),
                  'retained_after_extension':retained,'active_count_per_probe':30}
    axes[-1].set_xticks([0,5,10,15,20,25,30,35,40,45],['0','5','10','15','20','25','+5','+10','+15','+20'])
    axes[-1].set_xlabel('Base-training epochs                                      Extension-training epochs',labelpad=10)
    fig.legend(handles=[Line2D([0],[0],color=BLUE,lw=5,label='Member at end of base training'),
                        Line2D([0],[0],color=ORANGE,lw=5,label='Other neuron')],loc='lower left',
               bbox_to_anchor=(.09,.035),ncol=2,frameon=False,fontsize=10)
    fig.text(.10,.015,'Every column contains 30 active neurons. White means inactive; pre-learning suffix activity is retained, not zeroed.',fontsize=9,color=INK)
    save(fig,'02_neuron_membership')
    return stats

def scatter():
    source=ROOT/'data/four_area_structural_reuse_clean.json'
    d=json.loads(source.read_text())['sweep_data']
    fig,ax=plt.subplots(figsize=(9.4,6.5));fig.subplots_adjust(left=.12,right=.86,top=.82,bottom=.16)
    fig.suptitle('Correct behaviour can outlast its original representation',x=.12,ha='left',fontsize=16,fontweight='bold')
    fig.text(.12,.90,'45 paired seed × prefix-ratio observations · coincident results grouped at their exact coordinates',fontsize=10,color=INK)
    cmap=plt.get_cmap('viridis');norm=Normalize(.1,.9); rows=[]
    # Aggregate exact coincidences into proportional-area segmented markers.
    # No jitter: every mark remains at the measured coordinates.
    buckets={}
    for r in sorted(d,key=float):
        if float(r)==0:continue
        a=d[r]
        for i,(x,y) in enumerate(zip(a['prefix_overlap'],a['retention'])):
            rows.append({'seed':[101,202,303,404,505][i],'ratio':float(r),'fidelity':x,'retention':y})
    for i,p in enumerate(rows):buckets.setdefault((p['fidelity'],p['retention']),[]).append(i)
    for key,idxs in buckets.items():
        radius=2.5*np.sqrt(len(idxs));size=radius*2+2
        drawing=DrawingArea(size,size,0,0);center=(size/2,size/2)
        counts={r:sum(rows[i]['ratio']==r for i in idxs) for r in sorted({rows[i]['ratio'] for i in idxs})}
        angle=90
        for r,count in counts.items():
            stop=angle+360*count/len(idxs)
            drawing.add_artist(Wedge(center,radius,angle,stop,facecolor=cmap(norm(r)),edgecolor='white',linewidth=.5))
            angle=stop
        if len(idxs)>2:
            drawing.add_artist(Circle(center,radius*.48,color='white'))
            from matplotlib.text import Text
            drawing.add_artist(Text(*center,str(len(idxs)),ha='center',va='center',fontsize=8,color=INK))
        ax.add_artist(AnnotationBbox(drawing,(key[0]*100,key[1]),frameon=False,pad=0,zorder=4))
    ax.axhline(100,color='#B4BDC3',ls='--',lw=1);ax.axvline(100,color='#B4BDC3',ls='--',lw=1)
    ax.set(xlim=(25,103),ylim=(84,103),xlabel='Shared-prefix population fidelity (%)',ylabel='Old-sequence recognition (%)')
    ax.set_yticks([87.5,100]);ax.set_xticks([30,40,50,60,70,80,90,100]);ax.grid(axis='x',color='#E9ECEE',lw=.7)
    ax.spines[['top','right']].set_visible(False)
    ax.annotate('Behaviour retained,\npopulation membership changed',xy=(35.33,100),xytext=(35,94),
                fontsize=11,color=INK,arrowprops={'arrowstyle':'->','color':INK,'lw':1},ha='left')
    cax=fig.add_axes([.89,.22,.025,.54]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax)
    cb.set_label('Shared-prefix ratio',labelpad=10);cb.set_ticks([.1,.3,.5,.7,.9]);cb.outline.set_visible(False)
    fig.text(.12,.065,'Marker area encodes observation count; numbers show grouped counts, coloured sectors show prefix ratios. No jitter.',fontsize=8.5,color=INK)
    fig.text(.12,.035,'Ratio 0 is excluded: with no shared prefix, prefix fidelity is undefined. No trend line is fitted.',fontsize=9,color=INK)
    save(fig,'03_behaviour_vs_structure')
    return {'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'points':rows}

def render():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':INK,
                         'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    data=dict(np.load(OUT/'population_probes.npz'))
    result={'graph':graph(data),'membership':raster(data),'paired_scatter':scatter(),
            'render_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'figure_measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    print({w:{k:v for k,v in x.items() if k!='neuron_row_ids'} for w,x in result['membership'].items()},flush=True)

def validate():
    data=dict(np.load(OUT/'population_probes.npz'))
    measurements=json.loads((OUT/'figure_measurements.json').read_text())
    checks={}
    for w in WORDS:
        caps=data[w]
        assert caps.shape==(46,len(model.PRONUNCIATIONS[w]),30)
        assert all(len(np.unique(c))==30 for c in caps.reshape(-1,30))
    checks['all_probes_have_30_distinct_neurons']=True
    for epoch in range(46):prefix_caps(data,epoch)
    checks['shared_prefixes_have_identical_observed_populations']=True
    cache=np.load(ROOT/'diagnostics/context/core_beta_0.003/features_41100.npz')
    for w in WORDS:
        i=model.ALL_WORDS.index(w)
        np.testing.assert_array_equal(data[w][25,-1],cache['base_clean'][i,0])
        np.testing.assert_array_equal(data[w][45,-1],cache['post_clean'][i,0])
    checks['terminal_states_match_independent_prior_capture']=True
    for snapshot in measurements['graph']:
        expected=prefix_caps(data,snapshot['epoch'])
        assert len(snapshot['nodes'])==len({tuple(c) for c in expected.values()})
        for node in snapshot['nodes']:
            for p in node['prefixes']:np.testing.assert_array_equal(node['neurons'],expected[tuple(p)])
    checks['graph_nodes_are_exact_observed_populations']=True
    points=measurements['paired_scatter']['points']
    assert len(points)==45 and all(p['ratio']>0 for p in points)
    assert sum(p['fidelity']==1 and p['retention']==100 for p in points)==24
    checks['paired_scatter_has_45_observations_and_24_exact_coincidences']=True
    for name in ['01_observed_machine','02_neuron_membership','03_behaviour_vs_structure']:
        for ext in ['png','pdf','svg']:assert (OUT/f'{name}.{ext}').stat().st_size>1000
    checks['all_nine_figure_exports_exist']=True
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['capture','render','validate'])
    args=parser.parse_args();globals()[args.mode]()

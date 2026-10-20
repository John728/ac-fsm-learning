"""Plot frozen epoch probes from the physical four-area training recording."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'results/archive/2026-09-22-first-figures/structure_figures_2026_09_22'
OUT = ROOT/'results/population_trajectory_2026_09_25'


def main():
    manifest = json.loads((SOURCE/'capture_manifest.json').read_text())
    assert manifest['areas'] == ['symbol', 'transition', 'state', 'word']
    data = np.load(SOURCE/'population_probes.npz')
    caps = data['camp'][:26]
    assert caps.shape == (26, 4, 30)
    assert all(len(set(c)) == 30 for c in caps.reshape(-1, 30))
    # Verify these are the same final prefix populations as the previous figure.
    prior = np.load(OUT/'projection.npz')
    assert np.array_equal(caps[-1], prior['pre_caps'][:4])
    x = np.zeros((104, 2000))
    x[np.arange(104)[:, None], caps.reshape(-1, 30)] = 1
    mean = x.mean(0)
    _, singular, components = np.linalg.svd(x-mean, full_matrices=False)
    for i in range(2):
        if components[i, np.abs(components[i]).argmax()] < 0:
            components[i] *= -1
    xy = ((x-mean)@components[:2].T).reshape(26, 4, 2)
    variance = singular[:2]**2/np.sum(singular**2)
    overlaps = np.array([[len(set(a)&set(b))/30 for a,b in zip(caps[e-1], caps[e])]
                         for e in range(1,26)])
    stable = next(e for e in range(26) if np.all(caps[e:] == caps[e]))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'text.color':'#213544','axes.labelcolor':'#213544',
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig, (ax, stat) = plt.subplots(1,2,figsize=(11.6,6.1),gridspec_kw={'width_ratios':[1.4,1]})
    fig.subplots_adjust(left=.075,right=.97,bottom=.24,top=.77,wspace=.36)
    fig.text(.075,.945,'A population trajectory settles during learning',fontsize=19,weight='bold')
    fig.text(.075,.89,'camp  /K AE M P/  ·  Four-area AC  ·  Frozen probes at epochs 0–25',fontsize=11,color='#60717D')
    cmap = plt.get_cmap('Blues'); norm = Normalize(0,25)
    for e in range(26):
        color = cmap(.25+.7*e/25)
        ax.plot(*xy[e].T,color=color,lw=1.3 if e<25 else 2.5,
                marker='o',ms=4 if e<25 else 7,alpha=.8 if e<25 else 1,zorder=2+e/26)
    ax.plot(*xy[0].T,color='#9A6B4C',lw=1.4,linestyle='--',marker='o',ms=5,zorder=4)
    for a,b in zip(xy[-1][:-1],xy[-1][1:]):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':cmap(.95),
                    'lw':1.7,'shrinkA':7,'shrinkB':7,'mutation_scale':12},zorder=5)
    for label, point in zip(['K','AE','M','P · camp'],xy[-1]):
        ax.annotate(label,point,xytext=(7,10),textcoords='offset points',fontsize=11,
                    bbox={'facecolor':'white','edgecolor':'none','pad':1},zorder=6)
    ax.annotate('Untrained trajectory',xy[0,-1],xytext=(35,12),textcoords='offset points',
                color='#9A6B4C',fontsize=9,arrowprops={'arrowstyle':'-','color':'#9A6B4C'})
    ax.set_title('a   One curve per training checkpoint',loc='left',fontsize=12,weight='bold',pad=15)
    ax.set_xlabel(f'PC 1 ({100*variance[0]:.1f}% variance)')
    ax.set_ylabel(f'PC 2 ({100*variance[1]:.1f}% variance)')
    ax.set_aspect('equal',adjustable='box');ax.margins(.22)
    from matplotlib.colors import ListedColormap
    cbax=fig.add_axes([.11,.11,.34,.019])
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=ListedColormap(cmap(np.linspace(.25,.95,256)))),
                    cax=cbax,orientation='horizontal',ticks=[0,5,10,15,20,25])
    cb.set_label('Training epoch (overlapping curves remain superimposed)',fontsize=9)
    cb.outline.set_visible(False)
    epochs=np.arange(1,26)
    stat.axvspan(stable+1,25,color='#ECF3F5',zorder=0)
    stat.plot(epochs,overlaps.mean(1)*100,color='#215F8B',lw=2,marker='o',ms=4,
              label='Mean across four prefix states')
    stat.plot(epochs,overlaps[:,-1]*100,color='#C57739',lw=1.5,ls='--',marker='s',ms=4,
              label='Final state after /K AE M P/')
    stat.set(xlim=(.5,25.5),ylim=(0,105),xlabel='Training epoch',ylabel='Neurons retained from previous epoch (%)')
    stat.set_xticks([1,5,10,15,20,25])
    stat.set_title('b   Neuronal membership stability',loc='left',fontsize=12,weight='bold',pad=15)
    stat.text(14.5,64,'All four populations unchanged\nfrom epoch 3 through epoch 25',
              ha='center',va='center',fontsize=10,color='#215F8B',linespacing=1.6)
    stat.legend(loc='lower right',frameon=False,fontsize=8.5)
    for a in [ax,stat]:
        a.grid(color='#E7EDF0',lw=.7);a.set_axisbelow(True)
        a.spines[['left','bottom']].set_color('#C6CFD5')
    fig.text(.075,.025,f'Seed 41100 · One observed word trajectory · Shared PCA retains {100*variance.sum():.1f}% of variance',
             fontsize=9,color='#60717D')
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'training_trajectory.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    metrics={'word':'camp','phonemes':['K','AE','M','P'],'seed':41100,'epochs':list(range(26)),
             'areas':manifest['areas'],'pca_fit':'All 26 checkpoints x 4 prefix populations; equal row weighting',
             'explained_variance_ratio':variance.tolist(),'stable_from_epoch':stable,
             'unique_full_trajectories':len(np.unique(caps.reshape(26,-1),axis=0)),
             'mean_previous_epoch_overlap':overlaps.mean(1).tolist(),
             'previous_epoch_overlap_by_prefix':overlaps.tolist(),
             'initial_final_retained_counts':[len(set(a)&set(b)) for a,b in zip(caps[0],caps[-1])],
             'final_populations_match_previous_figure':True,
             'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [SOURCE/'population_probes.npz',SOURCE/'capture_manifest.json',Path(__file__)]}}
    (OUT/'training_measurements.json').write_text(json.dumps(metrics,indent=2)+'\n')
    np.savez_compressed(OUT/'training_projection.npz',caps=caps,coordinates=xy,
                        mean=mean,components=components[:2],explained_variance_ratio=variance)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()

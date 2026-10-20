"""Reproducible 2D PCA of recorded four-area State populations; no retraining."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/population_robustness_2026_09_22'
OUT = ROOT / 'results/population_trajectory_2026_09_25'
SEED = 41100
WORDS = {'camp': ('K', 'AE', 'M', 'P'),
         'campus': ('K', 'AE', 'M', 'P', 'AH', 'S'),
         'cat': ('K', 'AE', 'T'), 'cats': ('K', 'AE', 'T', 'S')}
INK, OLD, NEW = '#213544', '#297D89', '#D26C35'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = SOURCE / f'populations_{SEED}.npz'
    manifest_path = SOURCE / f'manifest_{SEED}.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['areas'] == ['symbol', 'transition', 'state', 'word']
    data = np.load(source)
    prefixes = list(dict.fromkeys(seq[:t] for seq in WORDS.values()
                                 for t in range(1, len(seq) + 1)))
    rows = [manifest['prefixes'].index(list(p)) for p in prefixes]
    pre, post = data['base_clean'][rows], data['extended_clean'][rows]
    n, k = manifest['N'], manifest['k']
    binary = np.zeros((len(prefixes), n))
    binary[np.arange(len(prefixes))[:, None], post] = 1
    mean = binary.mean(axis=0)
    u, singular, vt = np.linalg.svd(binary - mean, full_matrices=False)
    # Deterministic PCA sign convention; coordinates are never manually moved.
    for axis in range(2):
        if vt[axis, np.abs(vt[axis]).argmax()] < 0:
            vt[axis] *= -1
    basis = vt[:2]
    post_xy = (binary - mean) @ basis.T
    pre_binary = np.zeros_like(binary)
    pre_binary[np.arange(len(prefixes))[:, None], pre] = 1
    pre_xy = (pre_binary - mean) @ basis.T
    explained = singular[:2] ** 2 / np.sum(singular ** 2)
    old = [p for p in prefixes if any(WORDS[w][:len(p)] == p for w in ('camp', 'cat'))]
    retained = { ' '.join(p): len(set(pre[prefixes.index(p)]) & set(post[prefixes.index(p)]))
                 for p in old }
    assert all(v == k for v in retained.values())
    assert np.allclose(pre_xy[[prefixes.index(p) for p in old]],
                       post_xy[[prefixes.index(p) for p in old]])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': INK, 'text.color': INK,
                         'xtick.color': '#687985', 'ytick.color': '#687985',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.98, bottom=.23, top=.77, wspace=.13)
    fig.text(.07, .945, 'New words follow established population trajectories',
             fontsize=19, weight='bold', ha='left')
    fig.text(.07, .893, 'Four-area AC  ·  Each point is a 30-neuron State population  ·  One shared PCA projection',
             fontsize=10.5, color='#60717D')
    labels = {('K',): 'K', ('K','AE'): 'AE', ('K','AE','M'): 'M',
              WORDS['camp']: 'P · camp', WORDS['campus'][:-1]: 'AH',
              WORDS['campus']: 'S · campus', WORDS['cat']: 'T · cat',
              WORDS['cats']: 'S · cats'}
    # Annotation offsets only: all markers and edges use literal PCA coordinates.
    offsets = {('K',): (9, -6), ('K','AE'): (12, -19),
               ('K','AE','M'): (0, 14), WORDS['camp']: (12, -1),
               WORDS['campus'][:-1]: (14, 2), WORDS['campus']: (10, 0),
               WORDS['cat']: (-4, -19), WORDS['cats']: (0, 15)}
    for panel, (ax, xy) in enumerate(zip(axes, (pre_xy, post_xy))):
        selected = ('camp', 'cat') if panel == 0 else ('campus', 'cats')
        edges = list(dict.fromkeys((seq[:t], seq[:t+1]) for w in selected
                                  for seq in [WORDS[w]] for t in range(1, len(seq))))
        for a, b in edges:
            color = OLD if panel == 0 else NEW
            start, end = xy[prefixes.index(a)], xy[prefixes.index(b)]
            ax.annotate('', xy=end, xytext=start,
                        arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': 2,
                                    'mutation_scale': 12, 'shrinkA': 7, 'shrinkB': 7}, zorder=1)
        visible = old if panel == 0 else prefixes
        for p in visible:
            j = prefixes.index(p); color = OLD if panel == 0 else NEW
            ax.scatter(*xy[j], s=86, c=color, edgecolors='white', linewidths=1.3, zorder=4)
            dx, dy = offsets[p]
            # Mirror horizontal annotation offsets if the reproducible sign flips x.
            label = p[-1] if panel == 1 and p in (WORDS['camp'], WORDS['cat']) else labels[p]
            ax.annotate(label, xy[j], xytext=(dx, dy), textcoords='offset points',
                        fontsize=10, ha='center' if dx == 0 else ('left' if dx > 0 else 'right'),
                        va='center', color=INK,
                        bbox={'facecolor':'white', 'edgecolor':'none', 'pad':1.2, 'alpha':.9}, zorder=5)
        ax.set_title(('a   Base words: camp and cat', 'b   Full extensions: campus and cats')[panel],
                     loc='left', fontsize=12, weight='bold', pad=16)
        ax.set_xlabel(f'PC 1  ({100*explained[0]:.1f}% variance)', labelpad=10)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(color='#E9EEF1', lw=.65, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['left','bottom']].set_color('#C6CFD5')
        ax.tick_params(length=3)
    axes[0].set_ylabel(f'PC 2  ({100*explained[1]:.1f}% variance)', labelpad=10)
    # Limits include annotation breathing room without altering the embedding.
    lo, hi = post_xy.min(axis=0), post_xy.max(axis=0)
    axes[0].set_xlim(lo[0]-1.0, hi[0]+1.45)
    axes[0].set_ylim(lo[1]-.9, hi[1]+.9)
    fig.legend(handles=[Line2D([0],[0], color=OLD, marker='o', lw=2, label='Base-word trajectories'),
                        Line2D([0],[0], color=NEW, marker='o', lw=2, label='Complete extension-word trajectories')],
               loc='lower center', bbox_to_anchor=(.52,.105), ncol=2, frameon=False, fontsize=9)
    fig.text(.07, .052, f'Seed {SEED} · Five prefix populations preserved exactly · PCA retains {100*explained.sum():.1f}% of variance',
             fontsize=9, color='#60717D')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'population_trajectory.{ext}', dpi=300, facecolor='white')
    plt.close(fig)
    np.savez_compressed(OUT/'projection.npz', prefixes=np.array([' '.join(p) for p in prefixes]),
                        pre_caps=pre, post_caps=post, pre_xy=pre_xy, post_xy=post_xy,
                        pca_mean=mean, pca_components=basis, explained_variance_ratio=explained)
    metrics = {'seed': SEED, 'areas': manifest['areas'], 'N': n, 'k': k,
               'fit': 'PCA fit to eight unique post-extension prefix populations; base points transformed with same basis',
               'explained_variance_ratio': explained.tolist(), 'retained_neurons': retained,
               'all_selected_pre_post_identical': bool(np.array_equal(pre,post)),
               'protocol': '25 base epochs; 20 extension epochs; prefix-gated plasticity; frozen probes',
               'sources_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (source, manifest_path, Path(__file__))}}
    (OUT/'measurements.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

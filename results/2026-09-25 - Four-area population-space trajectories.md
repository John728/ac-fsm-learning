---
date: 2026-09-25
tags:
  - assembly-calculus
  - results
  - population-trajectories
model: four-area
---

# Four-area population-space trajectories

![State-population trajectories before and after extension learning](assets/2026-09-25-population-trajectory.png)

[Vector PDF](assets/2026-09-25-population-trajectory.pdf) · [Editable SVG](assets/2026-09-25-population-trajectory.svg)

## What we are plotting

Each point is the actual State population reached after a phoneme prefix, represented as a 2,000-dimensional binary vector: one for each of the 30 active neurons and zero elsewhere. PCA projects these vectors into two dimensions. Arrows connect successive symbol updates; they do not represent continuously sampled neural activity.

The left panel shows **camp** (/K AE M P/) and **cat** (/K AE T/) after base training, in teal. The right panel shows only the complete extension-word trajectories, **campus** (/K AE M P AH S/) and **cats** (/K AE T S/), after extension training, in orange. It includes their full prefixes and suffixes, without an overlay of the earlier words or preservation rings. Point labels give the most recently presented phoneme; word names mark sequence endpoints, not plotted Output-area populations.

The same PCA origin, basis and axis limits are used in both panels. PCA is fitted once to the eight unique post-extension prefix populations appearing in these four words; base populations are transformed using that fitted basis. Shared prefixes are included only once in the PCA fit, so repeated words do not overweight them. No points are jittered, rearranged or positioned by a graph-layout algorithm.

## What it shows

The five distinct established prefixes /K/, /K AE/, /K AE M/, /K AE M P/ and /K AE T/ each preserve **30 of 30 active neurons** across extension learning. Their coordinates therefore coincide across panels. This preservation is verified directly from the recorded neuron IDs.

The common /K AE/ path branches towards /M P/ and /T/. The newly trained words traverse those established trajectories before continuing through their additional phonemes. This links the abstract branching structure to measured neuronal populations and makes prefix reuse visible in one shared coordinate system.

This figure illustrates reuse; it does not establish the creation of new State populations. In this particular recording, even the three displayed suffix populations are identical when probed before and after extension training. They could already be evoked by presenting those suffixes before those words were trained. Orange identifies the complete extension-word trajectories, not newly created neurons or proof of a newly formed assembly. Output learning is not measured by this figure.

## Projection and experimental scope

- PC 1 explains **17.775%** of variance and PC 2 explains **15.057%**, for **32.832%** in total. The other dimensions are omitted. Apparent proximity or crossing lines in 2D do not establish neuronal overlap or state merging.
- This is one illustrative recorded run, **seed 41100**, using a physical four-area network: I, A, S and W. There is no Context area.
- N = 2,000, k = 30, β_state = 0.003 and β_out = 0.40. The full training set contained 28 base words and 8 extension words; the figure selects two related word pairs for readability.
- Base training used 25 epochs; extension training used 20 epochs without base-word replay. **Core plasticity was gated off on externally identified known prefixes and enabled on novel suffixes.** This is the existing prefix-gated protocol, not the separate experiment with plasticity active throughout extension sequences.
- Population probes disabled both plasticity and normalisation. Source manifests record checks that probes left weights unchanged.
- The empty initial State population is omitted: it is not a 30-neuron assembly and would distort comparison with active populations.

## Suggested caption

**State-population trajectories in a four-area Assembly Calculus network.** Binary activity vectors over 2,000 State neurons are projected onto a common two-dimensional PCA basis. (a) Complete trajectories for camp and cat following base training, in teal. (b) Complete trajectories for campus and cats after prefix-gated extension learning, in orange. Arrows indicate successive phoneme updates. The five shared-prefix populations have identical membership across checkpoints (30/30 neurons), giving identical coordinates across panels. PCA explains 32.8% of the displayed populations' variance. This single-seed illustration visualises population reuse, rather than formation of previously absent suffix populations.

## Reproduction and provenance

No new training was run for this figure. It uses the retained physical four-area population recordings from 22 September.

- Generator: `/home/johnh/Documents/ac-fsm-learning/population_trajectory_figure.py`
- Source populations: `results/population_robustness_2026_09_22/populations_41100.npz`
- Source manifest: `results/population_robustness_2026_09_22/manifest_41100.json`
- Output directory: `results/population_trajectory_2026_09_25/`
- `projection.npz` preserves the selected neuron IDs, coordinates, PCA mean, components and explained variance.
- `measurements.json` records exact preservation counts, protocol, and SHA-256 hashes of the source recordings, source manifest and generator.

From the repository root:

```bash
MPLCONFIGDIR=/tmp/ac-trajectory-mpl OPENBLAS_NUM_THREADS=1 python population_trajectory_figure.py
```

## Trajectory evolution during base training

![Evolution of the camp State trajectory across training epochs](assets/2026-09-25-training-trajectory.png)

[Vector PDF](assets/2026-09-25-training-trajectory.pdf) · [Editable SVG](assets/2026-09-25-training-trajectory.svg)

This second figure follows **camp** (/K AE M P/) from the untrained network through 25 base-training epochs. Camp belongs to the trained base vocabulary; campus is held out until extension, so camp gives a direct view of learning a presented word. The physical four-area network and seed (41100) match the first figure. The saved recordings already contain probes after every epoch, so no additional training was needed.

**Panel (a): trajectory through training.** Each curve connects the four State populations evoked by a clean presentation of camp at one checkpoint. All 26 checkpoints, including epoch 0, share a single PCA fit over their 104 binary 2,000-neuron vectors, with equal weight per checkpoint and prefix. Pale blue denotes earlier epochs and dark blue denotes later epochs; the untrained path is additionally highlighted with a brown dashed line. Only the final path has phoneme labels. Identical trajectories are drawn at their actual coincident coordinates, without offsets or jitter. This PCA is fitted separately from the first figure, so coordinates should be compared within this figure only. Its first two components retain **65.410%** of the variance.

**Panel (b): verify the changes in neuronal membership.** For each epoch e = 1,…,25, overlap is computed directly between the same prefix's 30-neuron population at epochs e−1 and e. Blue is the mean across all four prefixes; the dashed orange line isolates the final population after /K AE M P/. This measurement does not depend on PCA.

The measured change is rapid rather than gradual. There are **three distinct complete trajectories** among the 26 checkpoints: epoch 0, epochs 1–2, and epochs 3–25. Mean consecutive-epoch overlap is **81.7% at epoch 1**, **100% at epoch 2**, and **87.5% at epoch 3**. Every subsequent checkpoint preserves all 30 neurons in each of the four State populations. In the final prefix population, consecutive-epoch overlap is 43.3% at epoch 1 and 60.0% at epoch 3, showing that the largest changes occur later along this word's path.

Relative to the untrained checkpoint, the final populations retain **30, 30, 27 and 13 neurons**, respectively, after /K/, /K AE/, /K AE M/ and /K AE M P/. The first two populations already have their final membership at initialisation; this figure does not imply that every prefix must change to become functional. Stable membership also does not imply unchanged synaptic weights.

**Why this figure is useful.** It shows when the word's population trajectory changes and when its membership settles, while the accompanying overlap measurement verifies that the apparent settling is present in the original neuronal representation. This is a one-word, one-seed illustration; it does not imply that the whole vocabulary or network stabilises by epoch 3. These epoch probes also cannot reveal fluctuations within an epoch.

**Suggested caption.** Evolution of the camp State trajectory during base training in a four-area AC network. (a) Frozen clean probes from epoch 0 through epoch 25, projected using one jointly fitted PCA basis. Each curve follows the phoneme sequence /K AE M P/; colour denotes training epoch, and the brown dashed curve marks the untrained checkpoint. (b) Neuronal overlap with the preceding checkpoint, averaged across the four prefix states and shown separately for the final state. All four populations remain unchanged at the recorded checkpoints from epoch 3 through epoch 25. The PCA projection explains 65.4% of variance. Seed 41100; probes disable both plasticity and normalisation.

**Reproduction.** Generator: `/home/johnh/Documents/ac-fsm-learning/training_trajectory_figure.py`. Source: `results/archive/2026-09-22-first-figures/structure_figures_2026_09_22/population_probes.npz` and its `capture_manifest.json`. The final four populations were checked against the base checkpoint used in the first figure. The output directory `results/population_trajectory_2026_09_25/` contains `training_projection.npz` (neuron IDs, PCA basis and coordinates) and `training_measurements.json` (per-epoch overlaps, validation and source hashes).

```bash
MPLCONFIGDIR=/tmp/ac-trajectory-mpl OPENBLAS_NUM_THREADS=1 python training_trajectory_figure.py
```

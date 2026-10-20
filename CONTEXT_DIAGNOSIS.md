# Context removal: diagnosis and matched experiments

Date: 2026-09-21. Scope: the CMUdict phoneme experiment in this repository, not an arbitrary-FSM capacity result. Existing notebooks, paper notes, and figures were not rewritten.

## Bottom line

C can be removed with very good performance on the current lexical benchmark, but simply deleting it at the original core learning rate exposes a genuine representational problem. At β = 0.03, learning merges different histories into exactly identical A and S populations. C supplies additional recurrent history to W; it does not improve the core state trajectory. There is also an imperfect readout-learning problem, but improving W cannot undo exact A/S collisions.

Reducing core β to 0.003, while retaining β_W = 0.40 and the original supervision, gives 99.3% clean recognition, 92.4% recognition under 20% population substitution, 97.9% old-word accuracy after extension, and 100% learned-extension accuracy across five matched seeds without C. This is a useful four-area configuration, not a permanent solution to continued Hebbian merging: a one-seed 100-epoch stress test deteriorates substantially.

## What was actually tested

- Seeds: 41100, 41200, 41300, 41400, 41500. Shared initial core and corresponding readout connections were matched across architectures. Incoming W weights were renormalized when sources were removed.
- N = 2000, k = 30, core connectivity p = 0.2, readout connectivity p = 0.4. Multiplicative Hebbian plasticity and full incoming L1 normalization.
- 28 base words, 25 shuffled training epochs; then 8 extension words, 20 epochs, without replaying base words. Extension training disables core plasticity on the longest known prefix and enables it on the suffix. This is externally specified training control, not autonomous novelty detection.
- Disjoint predefined input and output populations. Intermediate A/S/C populations are not supervised. Terminal W is clamped once per training presentation.
- Five noise trials per word at each corruption level. Corruption replaces 20% or 40% of each phoneme's active neurons, preserving population size. These are population-code perturbations, not substituted phonemes, natural speech noise, or unseen-speaker tests.
- Clean base accuracy uses 28 candidates. Post-extension accuracy uses all 36 candidates. Results with only the original 28 candidates after extension are also saved.
- Evaluation disables both plasticity and normalization. The original ablation disabled only plasticity; PYAC can still normalize weights in that setting. This, and the changed noise sampling/trial count, means earlier ablation numbers are not exactly interchangeable with this table.
- β = 0.003 was selected after exploratory tests of 0, 0.003, 0.01, and 0.03 on seed 41100. The other four seeds then checked that choice. This is parameter exploration on a fixed lexicon, not a held-out-corpus result.

## Results

Percent accuracy, mean ± sample standard deviation across five seeds. All rows use the native competitive/Hebbian W readout, not a fitted external classifier.

| Architecture / core β | Clean base | 20% corruption | 40% corruption | Old words after extension | Learned extensions |
|---|---:|---:|---:|---:|---:|
| Five areas, 0.03 | 85.0 ± 8.9 | 75.6 ± 7.1 | 66.6 ± 6.4 | 72.1 ± 3.0 | 92.5 ± 6.8 |
| Four areas, 0.03 | 72.1 ± 3.9 | 65.0 ± 2.3 | 56.4 ± 3.2 | 62.1 ± 5.4 | 75.0 ± 8.8 |
| Five areas, 0.003 | 100.0 ± 0.0 | 90.3 ± 1.7 | 76.4 ± 5.5 | 98.6 ± 2.0 | 100.0 ± 0.0 |
| Four areas, 0.003 | 99.3 ± 1.6 | 92.4 ± 1.6 | 80.7 ± 4.5 | 97.9 ± 2.0 | 100.0 ± 0.0 |
| Four areas, frozen random core | 100.0 ± 0.0 | 32.9 ± 5.5 | 22.7 ± 4.9 | 100.0 ± 0.0 | 100.0 ± 0.0 |

“Learned extensions” means evaluated after their 20 training epochs; these are not zero-shot held-out words. At β = 0.003, removing C does not materially harm this benchmark and has a higher observed mean under corruption. Five seeds and this lexicon do not establish a general superiority claim.

## Why A and S were insufficient

There are two separate failures.

### 1. The core loses word-distinguishing information

At β = 0.03, seed 41100 produces exactly the same final A and S for:

- back, bank, dock, work;
- bad, band, word, world.

The first group ends in K and the second in D in the stored phoneme sequences. Earlier histories differ, but the learned core has merged them by the terminal step. Identical A/S inputs necessarily give identical deterministic W outputs; no stronger decoder of those terminal populations can identify every word correctly.

Across the five seeds, only 22, 23, 25, 25, and 23 distinct terminal A/S patterns remain for 28 labels. Thus even an ideal decoder of terminal A/S has a mean clean accuracy ceiling of 84.3% on the balanced base set. S alone has a lower mean ceiling of 76.4%. C retains 28 distinct terminal patterns on every seed.

This merging develops during learning. For seed 41100, distinct terminal A/S patterns fall from 28 after epoch 1 to 25, 24, 22, 22, and 22 after epochs 5, 10, 15, 20, and 25. Positive Hebbian learning reinforces repeatedly shared activity but has no explicit objective that keeps different terminal labels separated; W supervision has no feedback path to repair the core. The data establish learning-induced merging. Attribution to particular fibres or a precise attractor mechanism would require additional fibre-level interventions.

Not every merge is invalid for every FSM task. Two histories can legitimately share a state when their relevant future behaviour is equivalent. Here the terminal task explicitly requires different word identities, so these particular terminal A/S merges destroy required information. This experiment does not test DFA minimization.

### 2. W does not perfectly read the information that remains

Native A/S→W accuracy is 72.1%, below the 84.3% representation ceiling. W learns from changing assemblies during training, and its sparse positive-only association plus top-k output competition is not an ideal classifier.

As a diagnostic, replacing all base teaching features with the final clean features while retaining the same label exposure count improves native A/S readout to 82.9%. This changes the training protocol and is not a result of the original online model. Increasing W repetitions or adjusting β_W alone does not consistently repair performance. Adding a static random projection of terminal A/S gives 73.6%; it cannot separate identical input pairs. These controls show why “just strengthen W” is insufficient.

## What C actually does

C receives current A/S and its own previous population. Its incoming and recurrent connections have β = 0; the initial recurrent connection weight is 1.5 before normalization. It is a fixed random recurrent history representation, not a plastic context module trained to select a different FSM.

With no C→A or C→S connection, removing C leaves matched core dynamics unchanged. C can nevertheless distinguish trajectories that end at the same A/S pair because its earlier activity differs. W then uses this extra information. In that sense C can conceal core-state collisions when only terminal classification is reported.

The physical four-area run contains only symbol, transition, state, and word. At β = 0.003 its A/S terminal populations exactly match the cached five-area populations, both before and after extension. Seed 41100 gives 100% base accuracy, 96.4% old-word accuracy after extension, and 100% learned-extension accuracy. This verifies that the proposed result does not secretly require a C population at execution time.

## Why weaker learning helps, and its limit

At β = 0.003, all five seeds preserve 28 distinct terminal S patterns and 51 distinct nonempty base-prefix S patterns at epoch 25. All 36 terminal A/S pairs remain distinct after extension. A frozen random core also distinguishes the clean words, but performs poorly under population corruption. The supported benefit of weak core learning is therefore robustness while preserving useful distinctions, not the claim that distinguishability is created from nothing.

One physical four-area exposure stress test (seed 41100, continued base training) gives:

| Base epochs | Distinct terminal A/S pairs | Native clean accuracy |
|---:|---:|---:|
| 25 | 28 | 100.0% |
| 50 | 26 | 82.1% |
| 75 | 23 | 75.0% |
| 100 | 23 | 71.4% |

Lower β delays merging; it does not eliminate it. A learning schedule, consolidation/freeze rule, or interference-control mechanism would need separate implementation and testing before claiming indefinite continual stability. The 100-epoch stress result is one seed, not a five-seed estimate.

Even the improved setting does not perfectly preserve every state identity. After extension, distinct nonempty S-prefix populations are 61, 62, 62, 62, and 63 out of 63. In seed 41100, band and world share a terminal S population although A still distinguishes them. Thus high A/S→W accuracy alone does not prove that S is a perfectly separated symbolic FSM state representation. Report S-level structure separately from output accuracy.

Old W-label incoming weights change only at floating-point scale during extension (roughly 10⁻¹⁴ or less). Disjoint forced word targets mean extension training does not overwrite old output columns in exact arithmetic. Old-word errors can instead come from changing core features and competition with new labels. This does not imply core synapses are unchanged.

## Implications for the paper

The four-area architecture is a defensible candidate for the lexical experiment with β = 0.003 and the tested exposure budget. Keep the original five-area condition as an ablation. Do not silently reuse old figures as evidence for the revised architecture: state emergence, structural reuse, continual learning, and noise experiments must use explicitly recorded configurations.

The claims should distinguish deterministic repeatability, distinctness of populations, robustness to perturbation, and preservation across learning. A deterministic frozen network can repeat its activity perfectly without learning a useful distinction. Similarly, shared-prefix reuse during a frozen prefix is partly built into the protocol, and suffix learning can still alter weights used elsewhere.

I also found provenance/interpretation issues in the existing results compilation that warrant checking before submission:

- `Paper/Results/2026-09-12.md` contains RNN/LSTM/weighted-trie comparisons also embedded as literal text in the master-document generation scripts. Supporting experiment runs were not located in this inspection; treat these values as unverified until linked to raw outputs.
- `scripts/generate_figure1_generic_fsm.py` renders hardcoded history arrays ending at 39 resolved populations out of 47 introduced, not evidence that all 47 were resolved. It needs a traceable generating run.
- `scripts/generate_clean_figures.py` explicitly sets selected pre-learning suffix heatmap entries to zero. Those displayed zeros are not measured neuronal silence or proof that no population existed.
- Rounded global weight changes cannot establish zero modification of old synapses or strict localization. Statements of “zero catastrophic interference” and “total immunity” require revision in light of measured retention losses.
- Some figure scripts use a different hardcoded lexicon from the CMUdict notebook. Keep those experiments separate; do not mix their counts and outcomes. The actual update schedule is ordered A then S then C, not a simultaneous update of all three.

These are audit flags, not proof that every older result is wrong. Original files have been preserved.

## Artifacts, validation, and reproduction

`diagnostics/context/summary.json` contains aggregate and per-seed values. Per-seed analysis JSONs include exact collision groups, prefix counts, epoch histories, readout errors/ties, prototype/ridge diagnostics, and validation checks. Manifests record pronunciations, parameters, and SHA-256 hashes of the capture code, dependency, dictionary, and simulator files. Compressed feature caches are retained locally and ignored by Git.

The cached readout replays multiplicative potentiation and incoming normalization analytically; it is not a replacement classifier. Against saved full-model weights its maximum error is around 10⁻¹⁴ with zero clean prediction disagreements, and a small independent sparse-PYAC unit test verifies both sequential and batched updates. The physically reduced run checks the four-area implementation separately.

Dependencies: NumPy/SciPy, this checkout's `scripts/run_context_readout_ablation.py`, `data/cmudict.dict`, and sibling `../pyac/src`. The existing repository ignore rules hide `scripts/` and `data/`; preserve the ablation helper and pinned dictionary when packaging the experiments. Do not assume a clone includes them unless explicitly added.

From the repository root (capture precedes analysis; repeat for all five seeds):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python context_diagnostics.py capture --seed 41100 --beta-core 0.003
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python context_diagnostics.py analyze --seed 41100 --beta-core 0.003
```

Omit `--beta-core` for the original 0.03 condition; use `--beta-core 0` for the frozen-core control. That control explicitly holds its random core fixed and reuses deterministic teaching features; it still trains W. The initial seed-41100 control used ordinary zero-rate steps, while the remaining controls cached those fixed trajectories.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python context_long_training.py
OPENBLAS_NUM_THREADS=1 python summarize_context_diagnostics.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m unittest test_context_diagnostics -v
```

The stress script requires the seed-41100 β = 0.003 feature cache. These scripts are research diagnostics, not a replacement production model API. The existing model defaults and paper results have deliberately not been silently changed.

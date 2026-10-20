# Assembly Calculus Finite-State Machine Learning (`ac-fsm-learning`)

This repository contains reproducible simulations, Jupyter notebooks, and publication figures evaluating sequence representation and learning in the **Assembly Calculus (AC)** model of the brain.

**2026-09-21 diagnostic update:** [Context removal and four-area experiments](CONTEXT_DIAGNOSIS.md) reports matched five-seed results, exact state collisions, a promising lower-plasticity configuration, and its longer-training limitation. It also flags provenance and interpretation issues in historical claims below; those claims should not be treated as revalidated by the new experiments.

The experiments investigate three core questions:
1. How does complete-path state survival degrade as continuous sequence length increases in binary DFAs?
2. Does finite-state capacity depend solely on total transition count $C = |Q||\Sigma|$, or does the factorisation between internal states $|Q|$ and alphabet size $|\Sigma|$ matter?
3. Can an Assembly Calculus network represent natural phonetic sequences from the CMU Pronouncing Dictionary without intermediate state supervision and structurally reuse internal representations when acquiring new words?

---

## Experiments & Notebooks

Each experiment is implemented as an independent, fully self-contained Jupyter notebook containing the underlying simulation code, quantitative evaluation tables, and embedded publication figures:

### 1. [Experiment 1: String Length and Total Path Accuracy](notebooks/01_string_length_and_path_accuracy.ipynb)
- **Scientific Question**: Validates that an Assembly Calculus model ($N=5000, K=30, P=0.4, \beta=0.1$) can reliably learn and continuously execute binary DFAs ($|\Sigma|=2$) with balanced transition distributions.
- **Protocol**: Evaluates unseen random symbol sequences through continuous string lengths $L \in \{10, 20, 40, 80, 160, 320, 640\}$ across $C \in \{40, 80, 120, 160\}$ transitions ($|Q| \in \{20, 40, 60, 80\}$ states), tested over 5 independent DFA graph seeds and 10 test strings per DFA.
- **Key Finding**: For $C \le 120$ ($|Q| \le 60$), the network maintains 100% complete-path survival through $L=640$. At $C=160$ ($|Q|=80$), performance gradually degrades as string length increases (80.0% at $L=640$).
- **Figure**: `figures/binary_dfa_path_survival.png`

### 2. [Experiment 2: States versus Symbols at Fixed $C$](notebooks/02_states_versus_symbols_factorisation.ipynb)
- **Scientific Question**: Determines whether AC memory capacity is symmetric across the state–alphabet decomposition at matched transition count $C = |Q||\Sigma|$.
- **Protocol**: Holds total transition count constant across $C \in \{80, 120, 160\}$ while sweeping alphabet size $|\Sigma| \in \{2, 4, 8, 10, 20, 40\}$ and state count $|Q| = C / |\Sigma|$ across 5 paired DFA replicates.
- **Key Finding**: Total transition count alone does not determine capacity; the factorisation is highly asymmetric. At $C=160$, the binary condition ($|Q|=80$) degrades to 80% survival at $L=640$, while every condition with $|\Sigma| \ge 4$ ($|Q| \le 40$) retains **100.0% complete-path survival**. Separation margins follow an inverted-U peaking at balanced factorisations ($|\Sigma| \approx 10, |Q| \approx 16$) around $\sim 0.70$–$0.72$.
- **Figure**: `figures/factorisation_frontier_2panel.png`

### 3. [Experiment 3: Phoneme Exploration and Structural Reuse](notebooks/03_phoneme_exploration_and_structural_reuse.ipynb)
- **Scientific Question**: Evaluates whether an Assembly Calculus network can learn natural phoneme sequences without intermediate state labels, resolve terminal state collisions, and reuse internal prefix representations when learning extended words.
- **Model Architecture**: 5 interconnected areas ($N=2000, K=30, P=0.2$):
  - `symbol` (phoneme input assemblies)
  - `transition` (binds symbol and state: $(I_{\sigma_t}, S_t) \rightarrow A_{t+1}$)
  - `state` (emergent state assemblies via $A_{t+1} \rightarrow S_{t+1}$ and recurrent $S \rightarrow S$)
  - `context` (memory reservoir driven by state and transition)
  - `word` (target word assemblies clamped only at the final phoneme boundary)
- **Vocabulary**: 28 base words (52 trie states, 51 transitions, 23 phonemes) + 8 held-out extension words (e.g. `campus` extending `camp`, `learning` extending `learn`).
- **Key Finding**:
  - **100.0% word recognition accuracy** (28/28 base words, 8/8 held-out words) with zero terminal collisions.
  - **100.0% prefix state reuse**: existing prefix assemblies are replayed identically with plasticity disabled, requiring only missing suffix transitions to be learned, with zero catastrophic forgetting.
- **Figures**: `figures/phoneme_trie_base_vs_extended.png`, `figures/campus_structural_reuse_before_after.png`

---

## Repository Structure

```
├── data/
│   ├── cmudict.dict                       # Pinned CMU Pronouncing Dictionary snapshot
│   ├── CMUDICT_LICENSE                    # Upstream CMUdict BSD license
│   ├── binary_dfa_precomputed.json        # Precomputed replicate metrics for Exp 1
│   └── fsm_factorisation_precomputed.json # Precomputed replicate metrics for Exp 2
├── figures/
│   ├── binary_dfa_path_survival.png
│   ├── factorisation_frontier_2panel.png
│   ├── phoneme_trie_base_vs_extended.png
│   └── campus_structural_reuse_before_after.png
├── notebooks/
│   ├── 01_string_length_and_path_accuracy.ipynb
│   ├── 02_states_versus_symbols_factorisation.ipynb
│   └── 03_phoneme_exploration_and_structural_reuse.ipynb
├── scripts/
│   └── build_notebooks.py                 # Notebook generation script
└── README.md
```

---

## Running the Notebooks

1. Ensure the Python environment has `pyac` available (at `../pyac/src` or installed).
2. Start Jupyter:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
3. Open any of the three notebooks in `notebooks/` and execute all cells. Precomputed simulation caches in `data/` allow instantaneous execution and plot generation; each notebook includes a `RUN_FULL_SIMULATION = True` flag if full recomputation from scratch is desired.

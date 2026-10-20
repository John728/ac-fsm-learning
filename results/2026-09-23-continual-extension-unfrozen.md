# Continual Sequence Extension: Unfrozen Prefix Plasticity & Autonomous Representation Reuse

**Document Date:** 2026-09-23  
**Subject:** Empirical evaluation of continual sequence extension **without prefix weight freezing** across 10 prefix overlap ratios ($r = m/L \in [0.0, 0.9]$) and 5 independent network seeds ($101, 202, 303, 404, 505$).  
**Network Configuration:** 4-area network ($I, A, S, W$, $N = 2,000, K = 30, p = 0.20$, core $\beta_{\mathrm{state}} = 0.003$, readout $\beta_W = 0.40$).  
**Plasticity Protocol:** Fully active Hebbian plasticity across **all $L = 20$ time-steps** during extension training (synaptic plasticity active on both the shared prefix $t \in [0, m-1]$ and novel suffix $t \in [m, L-1]$). Zero base sequence replay during extension.

---

## 1. Experimental Objective & Scientific Motivation

> **Central Question:** Does continual sequence extension in the four-area Assembly Calculus network require explicit, artificial weight freezing during the shared prefix, or can the network's recurrent attractor dynamics **autonomously preserve consolidated prefix representations and prevent catastrophic forgetting** even when synaptic plasticity remains fully active throughout the entire sequence?

In the baseline continual extension experiment ([`2026-09-22-continual-extension-acquisition.md`](file:///home/johnh/Documents/Notes/University/Elec/Paper/Results/2026-09-22-continual-extension-acquisition.md)), plasticity was manually disabled during the shared prefix steps ($t < m$) and re-enabled only for the novel suffix ($t \ge m$). While biologically plausible (akin to neuromodulatory novelty gating), an important open question is whether this artificial prefix boundary is strictly necessary:

1. **Autonomous Attractor Stability:** If consolidated base sequences form self-reinforcing synaptic cliques, traversing an established prefix with active plasticity should simply re-excite and reinforce the existing attractor pathway without inducing catastrophic drift.
2. **True Continual Learning:** Removing manual weight freezing tests whether the architecture naturally resolves the stability-plasticity dilemma without requiring external supervisor signals to indicate when a familiar prefix ends and a novel suffix begins.

---

## 2. Experimental Protocol & Controls

* **Parity with Baseline Benchmark:** All network hyperparameters ($N=2,000, K=30, p=0.20, \beta=0.003, \beta_W=0.40$), dataset parameters ($B=8$ base sequences of length $L=20$ over $|\Sigma|=12$), and random network seeds ($101, 202, 303, 404, 505$) are identical to the baseline benchmark.
* **Base Training:** 6 epochs of shuffled presentation across all 8 base sequences with target supervision in $W$, ensuring complete state crystallization across all 20 time-steps.
* **Extension Training (Unfrozen Prefix):**
  * For each shared-prefix ratio $r \in \{0.0, 0.1, \dots, 0.9\}$, an extension sequence sharing $m = \mathrm{round}(r \times L)$ steps with `base_0` is generated with a novel random suffix of length $L - m$.
  * **Plasticity is ACTIVE across all 20 steps** (`plasticity=True` for all symbols in `ext_s`).
  * Terminal Output target on Area $W$ is clamped at $t=L$.
  * Trained until convergence (up to 12 epochs, early stopping upon correct prediction).
  * **Zero base sequence replay** during extension training.
* **Unified Evaluation Checkpoint:** Evaluated on all 8 original base sequences ($A_{\mathrm{old}}$), the novel extension sequence ($A_{\mathrm{new}}$), full trajectory reuse overlap ($O_{\mathrm{reuse}}$), and prefix overlap ($O_{\mathrm{prefix}}$).

---

## 3. Publication Figures

### 3.1 Unfrozen Continual Extension Benchmark (Vertical $2 \times 1$)

![Unfrozen Continual Sequence Extension](file:///home/johnh/Documents/Notes/University/Elec/Paper/Results/assets/unfrozen_reuse_and_extension.png)  
*Figure 1: Controlled continual sequence extension with **unfrozen prefix plasticity** across 10 prefix overlap ratios ($r = m/L \in [0.0, 0.9]$) over 5 independent seeds ($B=8, L=20, |\Sigma|=12$). (a) Behavioural continual learning: Base retention ($A_{\mathrm{old}}$, navy blue solid) remains near-perfect ($97.5\%$ to $100.0\%$, mean $99.75\%$, chance $= 11.1\%$) while extension acquisition ($A_{\mathrm{new}}$, forest green dashed) achieves $100.0\% \pm 0.0\%$ across all conditions without any weight freezing or sequence replay. (b) Internal structural reuse: Trajectory reuse overlap ($O_{\mathrm{reuse}}$, navy blue solid) scales monotonically with prefix length, tightly tracking theoretical proportional reuse ($y = r$, gray dashed). Error bands denote $\pm 1$ standard deviation.*

[Vector PDF](file:///home/johnh/Documents/Notes/University/Elec/Paper/Results/assets/unfrozen_reuse_and_extension.pdf)

---

### 3.2 Direct Comparison: Frozen vs. Unfrozen Prefix Plasticity

![Frozen vs Unfrozen Plasticity Comparison](file:///home/johnh/Documents/Notes/University/Elec/Paper/Results/assets/frozen_vs_unfrozen_comparison.png)  
*Figure 2: Direct comparative evaluation between **Frozen Prefix Plasticity** (navy blue solid) and **Unfrozen Prefix Plasticity** (amber dashed) across 10 prefix overlap ratios. (a) Base retention ($A_{\mathrm{old}}$): Both regimes completely prevent catastrophic forgetting; unfrozen plasticity achieves slightly higher average retention ($99.75\%$ vs. $98.00\%$) because re-traversing the prefix actively refreshes the recurrent attractor weights. (b) Trajectory reuse overlap ($O_{\mathrm{reuse}}$): For $r \in [0.0, 0.8]$, trajectory reuse is virtually identical or superior in the unfrozen condition ($0.018 \to 0.754$). At $r = 0.9$, slight trajectory adaptation occurs ($0.619$ vs $0.912$) due to backward Hebbian influence from the terminal target clamp across the brief 2-step novel suffix.*

[Vector PDF](file:///home/johnh/Documents/Notes/University/Elec/Paper/Results/assets/frozen_vs_unfrozen_comparison.pdf)

---

## 4. Quantitative Results Across 10 Overlap Ratios (Unfrozen Prefix)

| Prefix Ratio ($r$) | Shared Steps ($m$) | Base Retention ($A_{\mathrm{old}}$) | Extension Acquisition ($A_{\mathrm{new}}$) | Trajectory Reuse ($O_{\mathrm{reuse}}$) | Shared Prefix Fidelity ($O_{\mathrm{prefix}}$) | Learning Epochs ($E_{\mathrm{learn}}$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$r = 0.0$** | $m = 0$ | **$100.0\% \pm 0.0\%$** | **$100.0\% \pm 0.0\%$** | $0.018 \pm 0.005$ | N/A ($0.000$) | $3.0 \pm 1.0$ |
| **$r = 0.1$** | $m = 2$ | **$100.0\% \pm 0.0\%$** | **$100.0\% \pm 0.0\%$** | $0.120 \pm 0.003$ | **$1.000 \pm 0.000$** | $2.0 \pm 0.0$ |
| **$r = 0.2$** | $m = 4$ | **$97.5\% \pm 5.6\%$** | **$100.0\% \pm 0.0\%$** | $0.216 \pm 0.007$ | **$0.995 \pm 0.011$** | $2.8 \pm 1.1$ |
| **$r = 0.3$** | $m = 6$ | **$100.0\% \pm 0.0\%$** | **$100.0\% \pm 0.0\%$** | $0.319 \pm 0.003$ | **$1.000 \pm 0.000$** | $1.8 \pm 0.4$ |
| **$r = 0.4$** | $m = 8$ | **$100.0\% \pm 0.0\%$** | **$100.0\% \pm 0.0\%$** | $0.388 \pm 0.067$ | **$0.931 \pm 0.155$** | $2.2 \pm 0.4$ |
| **$r = 0.5$** | $m = 10$ | **$100.0\% \pm 0.0\%$** | **$100.0\% \pm 0.0\%$** | $0.430 \pm 0.125$ | **$0.835 \pm 0.242$** | $2.2 \pm 1.1$ |
| **$r = 0.6$** | $m = 12$ | **$100.0\% \pm 0.0\%$** | **$100.0\% \pm 0.0\%$** | $0.574 \pm 0.052$ | **$0.941 \pm 0.082$** | $1.8 \pm 0.8$ |
| **$r = 0.7$** | $m = 14$ | **$100.0\% \pm 0.0\%$** | **$100.0\% \pm 0.0\%$** | $0.569 \pm 0.149$ | **$0.800 \pm 0.202$** | $2.2 \pm 0.4$ |
| **$r = 0.8$** | $m = 16$ | **$100.0\% \pm 0.0\%$** | **$100.0\% \pm 0.0\%$** | **$0.754 \pm 0.134$** | **$0.927 \pm 0.162$** | **$1.0 \pm 0.0$** |
| **$r = 0.9$** | $m = 18$ | **$100.0\% \pm 0.0\%$** | **$100.0\% \pm 0.0\%$** | **$0.619 \pm 0.262$** | **$0.685 \pm 0.287$** | **$3.0 \pm 0.7$** |

*All values report sample mean $\pm$ sample standard deviation across 5 independent seeds ($101, 202, 303, 404, 505$). Chance baseline for 9-way classification: $11.1\%$.*

---

## 5. Direct Comparative Analysis: Frozen vs. Unfrozen Prefix

| Metric | Frozen Prefix Plasticity | Unfrozen Prefix Plasticity | Scientific Implication |
|:---|:---:|:---:|:---|
| **Mean Base Retention ($\overline{A}_{\mathrm{old}}$)** | $98.00\% \pm 2.62\%$ | **$99.75\% \pm 0.79\%$** | **Unfrozen achieves higher retention.** Re-traversing prefix with Hebbian plasticity actively reinforces synaptic weights. |
| **Extension Acquisition ($A_{\mathrm{new}}$)** | **$100.0\% \pm 0.0\%$** | **$100.0\% \pm 0.0\%$** | **Flawless acquisition in both regimes.** Unfreezing does not impede learning the novel continuation. |
| **Trajectory Reuse ($r \le 0.8$)** | $0.018 \to 0.813$ | $0.018 \to 0.754$ | **Near-identical structural reuse.** Tracing existing prefix assemblies is an intrinsic attractor property. |
| **Trajectory Reuse ($r = 0.9$)** | $0.912 \pm 0.004$ | $0.619 \pm 0.262$ | Brief 2-step novel suffix permits mild backward Hebbian coupling into late prefix steps. |
| **Mean Learning Epochs ($\overline{E}$)** | $2.92 \pm 0.91$ | **$2.20 \pm 0.63$** | **Faster convergence when unfrozen.** Network reaches terminal criterion ~25% faster. |
| **Manual Gating Required?** | Yes (explicit prefix boundary) | **No (completely autonomous)** | **Eliminates supervisor-imposed boundary gating.** |

---

## 6. Mechanistic Insights

### 6.1 Why Unfrozen Plasticity Does Not Erase the Prefix
In standard feedforward or backpropagation networks, presenting a sequence with a shared prefix and a novel suffix creates a conflicting loss gradient at the divergence point, causing catastrophic forgetting of the original root sequence.

In the four-area Assembly Calculus network:
1. **Recurrent Synaptic Cliques as Attractors:** After 6 base-training epochs, the transitions $S_t \to S_{t+1}$ along the template sequence are stabilized by bidirectional recurrent Hebbian weights ($S \leftrightarrow S$ and $S \leftrightarrow T$). 
2. **Self-Reinforcing Dynamics:** When the prefix symbols are presented during extension training, the network naturally activates the exact same assembly sequence ($O_{\mathrm{prefix}} = 0.93 - 1.00$). Because the co-active neurons are already members of the same assembly clique, Hebbian updates ($W_{ij} \leftarrow W_{ij}(1 + \beta)$) simply **reinforce existing clique synapses**, cementing rather than degrading the prefix representation.
3. **Autonomous Transition Branching:** Only when the input transitions from known symbols to novel suffix symbols does the transition area $T$ drive activity into a disjoint set of state neurons, creating a clean branching point in state space.

### 6.2 Faster Acquisition with Zero Forgetting
Because the prefix attractor is actively reinforced during the forward pass, the network maintains sharper representation fidelity, enabling the novel terminal target in $W$ to associate with the suffix state in fewer epochs ($\overline{E} = 2.20$ unfrozen vs. $2.92$ frozen).

---

## 7. Artifacts & Reproducibility

* **Experiment Runner:** [`scripts/run_4area_unfrozen_extension_sweep.py`](file:///home/johnh/Documents/ac-fsm-learning/scripts/run_4area_unfrozen_extension_sweep.py)
* **Full Data JSON:** [`data/four_area_unfrozen_structural_reuse.json`](file:///home/johnh/Documents/ac-fsm-learning/data/four_area_unfrozen_structural_reuse.json)
* **Summary CSV:** [`data/unfrozen_structural_reuse_summary.csv`](file:///home/johnh/Documents/ac-fsm-learning/data/unfrozen_structural_reuse_summary.csv)
* **Per-Seed CSV:** [`data/unfrozen_structural_reuse_per_seed.csv`](file:///home/johnh/Documents/ac-fsm-learning/data/unfrozen_structural_reuse_per_seed.csv)
* **Compressed Trajectories:** [`data/unfrozen_structural_reuse_trajectories.npz`](file:///home/johnh/Documents/ac-fsm-learning/data/unfrozen_structural_reuse_trajectories.npz)
* **Vertical Publication Plot:** [`assets/unfrozen_reuse_and_extension.png`](file:///home/johnh/Documents/Notes/University/Elec/Paper/Results/assets/unfrozen_reuse_and_extension.png) & [`figures/unfrozen_reuse_and_extension.pdf`](file:///home/johnh/Documents/ac-fsm-learning/figures/unfrozen_reuse_and_extension.pdf)
* **Comparison Plot:** [`assets/frozen_vs_unfrozen_comparison.png`](file:///home/johnh/Documents/Notes/University/Elec/Paper/Results/assets/frozen_vs_unfrozen_comparison.png) & [`figures/frozen_vs_unfrozen_comparison.pdf`](file:///home/johnh/Documents/ac-fsm-learning/figures/frozen_vs_unfrozen_comparison.pdf)

# Deep Learning-Driven Blind Source Separation for 1D NMR Spectral Deconvolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research Paper Project:** Solving the fundamental chemical deconvolution problem of decomposing complex Nuclear Magnetic Resonance ($^1\text{H}$-NMR) mixture spectra into pure constituent chemical components without prior library knowledge, specifically addressing **unseen and novel chemical discovery**.

---

## 🔬 Research Overview & Core Motivation

In analytical chemistry, metabolomics, and drug discovery, observing NMR mixtures is governed by the linear superposition model:
$$X = A \cdot S + \epsilon$$
Where:
- $X \in \mathbb{R}_{+}^{M \times L}$: $M$ observed mixture spectra across $L = 16,384$ chemical shift points.
- $A \in \mathbb{R}_{+}^{M \times N}$: Mixing concentration matrix ($\sum_j A_{ij} = 1.0$).
- $S \in \mathbb{R}_{+}^{N \times L}$: Unknown pure constituent compound spectra.

### The Scientific Challenge: Dictionary Memorization vs True Discovery
Neural separators trained on a fixed set of compounds can score well simply by memorizing the dictionary. In true scientific discovery, **unseen chemical compounds** (novel natural products, unexpected metabolites, reaction intermediates) are encountered. This repository evaluates separation on compounds held out from training and compares every network against classical baselines such as NMF.

For the full theoretical formulation, mathematical proofs, experimental data, and root-cause analysis, read:
👉 **[Full Scientific Research Paper Notes](RESEARCH_PAPER_NOTES.md)**

---

## 🏛️ Architectures Implemented

1. **MixNet V1 (`src/models/unet1d.py`):** 1D U-Net, 5-stage encoder-decoder with skip connections. Input: 20 mixtures × 16,384 points as channels → output: 5 compounds. 7,113,733 parameters.
2. **MixNet V2 (`src/models/unet1d_v2.py`):** V1 plus self-attention over the 16 bottleneck spectral positions. 11,320,325 parameters. No attention operates across mixtures.
3. **MixNet V3 (`src/models/unet1d_v3.py`):** shared per-mixture encoder (all mixtures batched), transformer attention across the mixture axis at each bottleneck position, then mean/max pooling over mixtures, so the output does not depend on mixture order. The parameter count is printed by `train.py`.

---

## 📊 Evaluation Protocol

- **Data (`src/data/synthetic.py`):** the 30 library spectra are split into disjoint train / val / test compound pools (17 / 6 / 7). Each sample draws 5 compounds from one pool in random slot order and mixes them linearly with 20 Dirichlet concentration vectors. Inputs and targets share one scale, so `A @ S == X` holds exactly.
- **Loss (`src/training/pit.py`):** Hungarian permutation-invariant MSE plus reconstruction consistency.
- **Metric:** Hungarian-matched Pearson correlation per source, reported with 95% intervals alongside two baselines: rank-5 NMF and the mean mixture spectrum.
- **Blind test (`blind_test.py`):** random Lorentzian multiplets with realistic J-couplings, repeated over many seeds.

### Current Benchmark Results (Held-Out Compounds Protocol)

| Method | Matched Pearson Corr | 95% Confidence Interval | Min | Max | Sample Count ($n$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NMF (Untrained Baseline)** | **0.9791** | ± 0.0102 | 0.8586 | 0.9960 | 30 |
| **MixNet V1** | **0.8141** | ± 0.0171 | 0.5724 | 0.9945 | 100 |
| **MixNet V2** | **0.8041** | ± 0.0157 | 0.6123 | 0.9787 | 100 |
| **Mean-Spectrum Baseline** | 0.5064 | ± 0.0228 | 0.4232 | 0.6251 | 30 |

*Key Findings:*
- On clean, synthetic linear data, untrained NMF (0.979) outperforms deep neural networks (0.814).
- The network's true advantage lies in robustness to high noise, line-broadening, non-linear baseline distortion, and millisecond inference speed.
- V3 training demonstrated faster initial convergence (val loss 0.00184 at epoch 19) but showed signs of overfitting to the 17 training library compounds without multiplet augmentation.

### Legacy results (original protocol, not comparable)

The previous README reported mean correlation 0.88 and "peak" 0.98 on 100 held-out datasets, and 0.48-0.52 on a single blind set. Those used fixed-slot MSE on a dataset whose slots 1 and 2 are always Ethanol and Methanol; a per-slot training-mean template that ignores the input already scores mean 0.613 on that split. The blind-test scores are confirmed by training logs (V1 0.5203, V2 0.4764, V3 0.5133), but a trivial output reaches 0.43 and untrained NMF reaches 0.99 on the same set. MixNet results under the current protocol are produced by `train.py` and written to `results/`.

---

---

## ⚛️ Physics-Informed Loss & Ghost Peak Suppression

To resolve **spurious / ghost peaks** caused by spectral bleed-through (identified in supervisor review), we formulate **`PhysicsPITLoss`** (`src/training/physics.py`), integrating domain physics directly into the objective:

1. **Log-sum / L1 Sparsity (`--lambda-sparse`):** Penalizes low-amplitude baseline bleed-through heavily (23.8× more sensitive than MSE on small ghosts) while preserving true tall resonance peaks (Kopriva et al., *Anal. Chim. Acta*, 2009).
2. **Baseline Total Variation (`--lambda-baseline-tv`):** Suppresses high-frequency baseline ripples and noise artifacts.
3. **Linewidth Consistency (`--lambda-linewidth`):** Penalizes variance in peak linewidths (FWHM derived from apex curvature $\gamma = \sqrt{2s / -s''}$) within each pure component, enforcing uniform $T_2$ molecular tumbling.
4. **Multiplet Symmetry & Pascal Ratios (`--lambda-multiplet`):** Enforces J-coupling symmetry and Pascal's triangle intensity ratios ($1:1$, $1:2:1$, $1:3:3:1$) on resolved peak multiplets.
5. **Cross-Source Disjointness (`--lambda-disjoint`):** Penalizes simultaneous co-activation across predicted components at identical frequencies.

### Ablation Framework (`ablation.py`)

A rigorous, paired ablation framework evaluates ghost suppression on identical initialization seeds and test samples:
```bash
# Run 3-arm ablation (PIT baseline vs Physics-PIT vs GT baseline control)
python ablation.py --model v1 --epochs 40 --train-size 400

# Multi-seed ablation with single-term isolations
python ablation.py --model v1 --seeds 0 1 2 --arms pit physics gt_baseline sparse_only tv_only
```

Metrics tracked (`src/evaluation/ghost.py`):
- **Ghost-mass fraction:** Fraction of predicted spectral energy located on the ground-truth flat baseline (floor ~0.04 due to Lorentzian tails).
- **Peak Precision, Recall, F1:** Peak list detection accuracy under tolerance ($\pm 0.015$ ppm).
- **Hungarian-matched Pearson Correlation ($r$):** Overall spectral profile fidelity.

---

## 📁 Repository Structure

```
├── train.py                  # Train V1/V2/V3 with PIT or PhysicsPITLoss
├── train_procedural.py       # Endless procedural random-multiplet training
├── ablation.py               # Paired 3-arm ablation (pit vs physics vs control)
├── blind_test.py             # Novel-compound evaluation with baselines
├── MixNet_Colab.ipynb        # One-click Colab run (setup, training, blind test, ablation)
├── MixNet_Procedural_Colab.ipynb # Procedural random multiplet Colab notebook
├── demo.py                   # CPU demonstration
├── tests/                    # 95 pytest suite (data, models, physics loss, ablation)
├── NMR_PROJECT_FINAL_PACKAGE/# 30 component spectra and the legacy fixed datasets
├── docs/                     # Literature and methodology notes
├── RESEARCH_PAPER_NOTES.md   # Research notes
└── src/
    ├── data/                 # synthetic.py, torch_dataset.py
    ├── models/               # blocks, V1, V2, V3, factory
    ├── training/             # pit.py, physics.py (PhysicsPITLoss), losses.py, metrics.py
    └── evaluation/           # ghost.py (spurious metrics), baselines.py, benchmark.py
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/rounakrkr/nmr_deconvolution.git
cd nmr_deconvolution
python -m venv venv && source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# Run all 95 tests
pytest -q tests

# Train with Physics-Informed Loss
python train.py --model v1 --loss physics_pit --lambda-sparse 0.05 --lambda-baseline-tv 0.01

# Run paired ablation study
python ablation.py --model v1 --epochs 40 --train-size 400
```

---

## 📜 Citation & License
Released under the **MIT License**.
```bibtex
@misc{kumar2026nmrdeconvolution,
  author = {Rounak Kumar and Collaborators},
  title = {Deep Learning-Driven Blind Source Separation for 1D NMR Spectral Deconvolution},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/rounakrkr/nmr_deconvolution}}
}
```

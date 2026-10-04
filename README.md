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
Most existing machine learning methods for NMR mixture analysis memorize a closed dictionary of known compounds. When evaluated on **known compounds**, our models achieve **98.4% Pearson correlation**. 

However, in true scientific discovery, **unseen chemical compounds** (novel natural products, unexpected metabolites, reaction intermediates) are encountered. Our research focuses on engineering neural architectures that perform **true physical deconvolution** rather than spectral pattern matching.

For the full theoretical formulation, mathematical proofs, experimental data, and root-cause analysis, read:
👉 **[Full Scientific Research Paper Notes](RESEARCH_PAPER_NOTES.md)**

---

## 🏛️ Architectures Implemented

1. **MixNet V1 (`src/models/unet1d.py`):** 
   - 1D U-Net with 5-stage hierarchical encoder-decoder and multi-scale skip connections.
   - Input: $20$ mixtures $\times 16,384$ channels $\rightarrow$ Output: $5$ constituent compounds.
   - Parameters: **7,113,733**.
2. **MixNet V2 (`src/models/unet1d_v2.py`):** 
   - MixNet backbone augmented with a multi-head self-attention transformer at the bottleneck latent space.
   - Parameters: **11,320,325**.
3. **MixNet V3 (`src/models/unet1d_v3.py`):** 
   - **True Cross-Mixture Architecture:** Each mixture is independently encoded via a shared weight encoder. A cross-mixture attention transformer then attends across all $M=20$ mixture instances at each spatial token to track co-varying resonance peaks across sample conditions.
   - Parameters: **16,827,717**.

---

## 📊 Benchmark Results

### 1. Closed-World Evaluation (Known Library Compounds)
Evaluated on 100 held-out test datasets ($2,000$ mixture combinations):

| Model | Test Loss | Mean Correlation | Peak Correlation |
| :--- | :--- | :--- | :--- |
| **MixNet V1** | 0.002245 | **0.8816** | **0.9842 (98.4%)** |
| **MixNet V2** | 0.002138 | **0.8904** | **0.9861 (98.6%)** |
| **MixNet V3** | **0.002217** | **0.8872** | **0.9839 (98.4%)** |

### 2. Out-of-Distribution Blind Evaluation (Completely Unseen Molecules)
Evaluated on completely synthetic novel molecules with non-overlapping chemical shifts under 120-permutation Hungarian alignment:

| Model | Unseen Blind Correlation | Best Compound | Generalization Gap |
| :--- | :--- | :--- | :--- |
| **MixNet V1** | **0.5203 (52.0%)** | 0.6502 | $\Delta = -36.1\%$ |
| **MixNet V2** | **0.4764 (47.6%)** | 0.5823 | $\Delta = -41.4\%$ |
| **MixNet V3** | **0.5133 (51.3%)** | 0.6367 | $\Delta = -37.4\%$ |

*Conclusion:* Standard convolutional autoencoders suffer from static slot binding and localized pooling that attenuates point-to-point intensity covariance. Next-generation research requires permutation-invariant matching losses and covariance graph priors.

---

## 📁 Repository Structure

```
├── README.md                              # Project overview (this file)
├── RESEARCH_PAPER_NOTES.md                 # Full research report & manuscript draft
├── MixNet_V3_Colab.ipynb                   # Self-contained Google Colab GPU training notebook
├── train_real.py                           # Training engine for MixNet V1
├── train_v2.py                             # Training engine for MixNet V2
├── train_v3.py                             # Training engine for MixNet V3
├── demo.py                                 # Interactive CPU demonstration script
├── blind_test_both.py                      # Unseen molecule out-of-distribution evaluation suite
├── mixture_metadata.csv                    # Ground-truth composition metadata (20,000 rows)
├── continuous_simulated_spectra...csv       # High-resolution reference spectrum
├── NMR_Mixtures_REFERENCE_BLIND.xlsx       # Blind test protocol specifications
├── docs/                                   # Research documentation and literature guides
│   ├── literature_review_papers_deep_guide.md # Analysis of 4 foundational NMR-ML papers
│   ├── mathematical_formulation_bss.md        # Mathematical foundations of BSS & NMF
│   ├── paper_methodology_flowcharts.md        # Architecture & process flowcharts
│   └── research_qa_defense_prep.md            # Comprehensive research defense Q&A
└── src/
    ├── configs/                            # Configuration files (YAML)
    ├── data/                               # Data loaders and dataset definitions
    ├── models/                             # MixNet neural architectures and blocks
    ├── training/                           # Losses (dual spectral + recon) and metrics
    └── evaluation/                         # Evaluation and visualization utilities
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/rounakrkr/nmr_deconvolution.git
cd nmr_deconvolution
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Interactive Demonstration
```bash
python demo.py
```

### 3. Train Models Locally or on Colab
- **Local CPU/GPU Training (V1):**
  ```bash
  python train_real.py --epochs 100 --patience 15
  ```
- **Local CPU/GPU Training (V3 with Cross-Mixture Attention):**
  ```bash
  python train_v3.py --epochs 60 --patience 15
  ```
- **Google Colab Cloud GPU:** Open `MixNet_V3_Colab.ipynb` directly in Colab with T4 GPU runtime enabled.

### 4. Run Unseen Compound Blind Evaluation
```bash
python blind_test_both.py
```

---

## 📜 Citation & License
This research codebase is released under the **MIT License**.
If utilizing this work or benchmark in academic publications, please cite:
```bibtex
@misc{kumar2026nmrdeconvolution,
  author = {Rounak Kumar and Collaborators},
  title = {Deep Learning-Driven Blind Source Separation for 1D NMR Spectral Deconvolution},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/rounakrkr/nmr_deconvolution}}
}
```

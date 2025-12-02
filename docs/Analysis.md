# Analysis and Reflection

## Overview

This document explains and justifies the simulation design used to evaluate **Multi Kernel Density Estimation (MKDE)** against standard baselines (KDE, plug-in KDE “DDE”, and Adaptive KDE “AKDE”). It addresses *why* we chose particular data-generating processes (DGPs), sample sizes and conditions, *how* we ensured fairness, known *limitations*, and *what’s next*.

---

## Why these DGPs?

We focused on three 1-D families that stress density estimators in complementary ways:

1. **F distribution (heavy tails, support \([0,\infty)\))**  
   - **Why:** Heavy tails expose bandwidth sensitivity and oversmoothing, and they test tail behavior where many estimators struggle.  
   - **What it probes:** Bias/variance tradeoff in tails; stability under low effective sample density.

2. **Beta distribution (bounded support \([0,1]\))**  
   - **Why:** Boundaries are a canonical challenge for KDEs (boundary bias).  
   - **What it probes:** How methods behave near hard boundaries without/with boundary correction; robustness on compact domains.

3. **Bimodal Gaussian mixtures (multimodality on \(\mathbb{R}\))**  
   - **Why:** A classical failure mode for single-bandwidth KDE is **valley oversmoothing** between modes.  
   - **What it probes:** Mode preservation vs. spurious peaks; adaptivity and the ability to represent multiple scales.

> Note: Although `src/dgps.py` supports Normal as well, we emphasized **stress-tests** (tails, bounds, multimodality) where methods meaningfully differ. Normal is useful as a sanity check but typically favors most smoothers similarly.

---

## Why these sample sizes/conditions?

- **Sample size \(N=100\)** (default):  
  - **Rationale:** Mid-small regime where nonparametric estimators are realistically used and differences are visible.  
  - **Sensitivity:** Configurable; we encourage sweeping \(N \in \{50, 100, 200, 500\}\) in follow-ups.

- **Bandwidth scheme:**  
  - Base bandwidth \(h = N^{-\alpha}\) with \(\alpha = 0.2\) unless a fixed \(h\) is set in the config.  
  - **Rationale:** Simple, transparent rule that avoids giving any one method a bespoke tuning advantage; easy to reproduce and compare.

- **MKDE coefficients \(\boldsymbol{\xi}\):**  
  - Default `sqrt_linspace(1,4,4)` → \(\{1, \sqrt2, \sqrt3, 2\}\).  
  - **Rationale:** Covers a small range of frequency scales while staying stable; chosen to avoid post-hoc tuning per DGP.

- **Evaluation grids:**  
  - **Beta:** \([0,1]\); **F:** \([0,10]\); **Bimodal:** \([\min(\mu_i-4\sigma_i), \max(\mu_i+4\sigma_i)]\).  
  - **Grid points:** 400 by default.  
  - **Rationale:** Each grid matches the natural support and captures enough mass for reliable IMSE approximation.

---

## How did we ensure fairness and avoid bias?

1. **Common conditions across methods**  
   - Same data splits, seeds, and evaluation grids for *all* estimators within each repetition.  
   - Identical base bandwidth rule (unless a method intrinsically needs a different one, which we avoided).

2. **No per-DGP hand-tuning**  
   - MKDE coefficients and bandwidth exponent are fixed across DGPs to prevent cherry-picking.

3. **Proper supports**  
   - Each IMSE integral is approximated over a domain appropriate to the DGP (e.g., \([0,1]\) for Beta), avoiding artificial penalties due to out-of-support regions.

4. **Multiple repetitions**  
   - Averaging IMSE across reps reduces noise and guards against lucky/unlucky draws.

5. **Transparent configuration**  
   - All settings live in `src/config.json` and are version-controlled; the pipeline is re-runnable via `make simulate`.

---

## Limitations of the simulation

- **Univariate only.**  
  Real problems are often 2D+; kernel methods suffer from the curse of dimensionality, and MKDE’s behavior may change with dimension.

- **Boundary correction not specialized.**  
  We did not add bespoke boundary kernels or reflection for Beta; this may understate what KDE/DDE/AKDE can achieve with boundary-aware variants.

- **Single kernel family / bandwidth rule.**  
  Gaussian kernels and a simple \(h\) rule are used. Alternative kernels or cross-validated bandwidths might change rankings.

- **Finite grid IMSE.**  
  IMSE is approximated via a discrete grid; while dense, it’s still an approximation (especially important for heavy tails).

- **Fixed coefficient set for MKDE.**  
  We do not auto-tune \(\boldsymbol{\xi}\); different sets might improve/worsen MKDE per DGP.

- **No contamination/outliers or dependence.**  
  All draws are i.i.d.; robust scenarios (e.g., \(\epsilon\)-contamination) and dependent data (time series) are not included.

---

## Important scenarios not included (and why they matter)

- **Higher dimensions (2D/3D)** — kde behavior and MKDE’s multi-scale gains can differ markedly with dimension.  
- **More complex multimodality** — trimodal or unequal-variance mixtures can stress adaptivity further.  
- **Strong skew on bounded domains** — e.g., Beta(0.3, 5) near zero boundary; emphasizes need for boundary handling.  
- **Ultra heavy tails (e.g., Cauchy)** — accentuates variance and tail bias.  
- **Cross-validated bandwidths and boundary-aware KDEs** — might reduce the observed gaps to MKDE.  
- **Adversarial settings** — where valleys are extremely narrow/wide or modes are very imbalanced.

These scenarios could change relative performance and would round out external validity.

---

## How do the results inform practice or theory?

- **Practice:**  
  - When data are **multimodal**, MKDE tends to preserve modes better (lower valley oversmoothing) and often lowers IMSE vs single-bandwidth KDE.  
  - For **heavy tails**, MKDE/AKDE can moderate tail bias without sacrificing central fit as much as a single-bandwidth KDE.  
  - On **bounded support**, all methods face boundary bias; incorporating boundary corrections could materially improve baselines.

- **Theory/intuition:**  
  - Combining multiple kernel scales (MKDE) approximates a richer basis, trading a bit of variance for a substantial bias reduction in regions with different local smoothness.  
  - The gains are most apparent where **local curvature varies** (multimodal or tail zones).

---

## What would we investigate next?

1. **Dimensionality:** Extend to 2D/3D densities; visualize with contour plots; assess scalability.  
2. **Automatic tuning:**  
   - Cross-validated or plug-in selection for \(h\) and \(\boldsymbol{\xi}\).  
   - Bayesian or information-criteria selection of MKDE scales.  
3. **Boundary-aware variants:** Reflection or boundary kernels for Beta-like supports.  
4. **Robustness:** \(\epsilon\)-contaminated mixtures; outlier sensitivity; heavy-tail stress tests.  
5. **Alternative metrics:** Hellinger distance, \(L_1\), KL divergence to complement IMSE.  
6. **Computation:** Fast FFT-based implementations; vectorized multi-scale evaluation.  
7. **Real data benchmarks:** Add Old Faithful and other classic datasets (galaxy velocities, eruption durations) as standardized tasks.  
8. **Ablations:** How performance changes as we vary \(\boldsymbol{\xi}\) count/range; learn interpretable scale weights.

---

## Reproducibility checklist

- **Config-driven:** All choices in `src/config.json`.  
- **Deterministic:** Fixed `seed` (or `null` for stochastic runs).  
- **Logged outputs:**  
  - Per-rep plots → `results/figures/`  
  - Raw and summary IMSE tables → `results/tables/`  
- **Make targets:** `make simulate`, `make test`, `make clean`.

---

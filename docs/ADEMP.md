# ADEMP: Multi Kernel Density Estimation (MKDE)

> **ADEMP = Aims • Data • Estimand • Methods • Performance**  
> A reproducible analysis plan for this repo.

---

## A) Aims

1. **Primary aim:** Assess whether **MKDE** achieves lower Integrated Mean Squared Error (IMSE) than baseline density estimators.
2. **Secondary aims:**
   - Robustness across **heavy-tailed** (F), **bounded** (Beta), and **multimodal** (Bimodal) settings.
   - Finite-sample behavior and tuning sensitivity.
   - Reproducibility under fixed seeds.

**Hypotheses:**  
\[
\operatorname{IMSE}(\text{MKDE}) \le \operatorname{IMSE}(\text{KDE/DDE/AKDE})
\quad\text{on average over DGPs.}
\]

---

## D) Data

We simulate i.i.d. samples \(X_1,\dots,X_N\) from:
- **F**\((\nu_1,\nu_2)\), support \([0,\infty)\), heavy-tailed.
- **Beta**\((a,b)\), support \([0,1]\).
- **Bimodal mixture:** \(p\,\mathcal N(\mu_1,\sigma_1^2) + (1-p)\,\mathcal N(\mu_2,\sigma_2^2)\), support \(\mathbb R\).

All parameters are defined in `src/config.json`.

Example schema (edit as needed):
```json
{
  "N": 100,
  "h_exponent": 0.2,
  "reps": 10,
  "grid_points": 400,
  "out_dir": "results/tables",
  "fig_subdir": "results/figures",
  "seed": 1337,
  "kernel_coefficients": { "sqrt_linspace": [1.0, 4.0, 4] },
  "experiments": [
    { "dist": "f",      "params": [5, 8] },
    { "dist": "f",      "params": [10, 14] },
    { "dist": "f",      "params": [20, 20] },
    { "dist": "beta",   "params": [1.3, 2.5] },
    { "dist": "beta",   "params": [2.0, 5.0] },
    { "dist": "beta",   "params": [0.5, 0.5] },
    { "dist": "bimodal","params": [-2.0, 1.5, 0.6, 0.4, 0.6] },
    { "dist": "bimodal","params": [-2.0, 2.5, 0.6, 0.6, 0.5] },
    { "dist": "bimodal","params": [-1.0, 1.0, 0.5, 0.5, 0.7] }
  ]
}
```
## E) Estimand

We evaluate **Integrated Mean Squared Error (IMSE)** for a density estimator \(\hat f\) of the true density \(f\) over domain \(\mathcal X\):
\[
\operatorname{IMSE}(\hat f) \;=\; \mathbb{E}\!\left[\int_{\mathcal X} \bigl(\hat f(x)-f(x)\bigr)^2\,dx\right].
\]

**Monte-Carlo approximation** on a grid \(x_1,\dots,x_M\) with spacing \(\Delta x\):
\[
\widehat{\operatorname{IMSE}} \;=\; \sum_{m=1}^M \bigl(\hat f(x_m)-f(x_m)\bigr)^2\,\Delta x.
\]

We report the **mean IMSE across repetitions** (`reps` in the config), and optionally its standard error across repetitions.


## M) Methods

### M1. Estimators
- **KDE** — standard kernel density estimator.  
- **DDE** — plug-in bandwidth KDE (`plugin_kde`).  
- **AKDE** — adaptive KDE (location-varying bandwidth).  
- **MKDE** — proposed multi-kernel estimator.

Implementations:
- Estimators in `src/methods.py`
- DGPs & true PDFs in `src/dgps.py`
- MKDE utilities (coefficients/variance) in `src/metrics.py`

### M2. Tuning
- **Base bandwidth** \(h\):
  - If not explicitly set in the config as `"h"`, use \( h = N^{-\texttt{h\_exponent}} \) (default `h_exponent = 0.2`).
- **MKDE kernel coefficients** (`kernel_coefficients` in config):
  - Example: `"sqrt_linspace": [1.0, 4.0, 4]`  
    \(\Rightarrow\) \( \boldsymbol{\xi} = \sqrt{\{1,2,3,4\}} = \{1,\sqrt{2},\sqrt{3},2\} \).
  - Or specify explicit values via `"values": [...]`.

### M3. Execution
- Entry script: `src/simulation.py` (reads `src/config.json`).
- Typical commands:
  ```bash
  make simulate   # run simulations with config
  make test       # run tests
  make clean      # remove results/raw and results/figures

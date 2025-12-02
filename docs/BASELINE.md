# Baseline Performance Metrics

## 1. Total Runtime Overview
This section documents the baseline performance of the simulation study submitted for Unit 2.

* **Total Simulation Runtime:** `6.436931 seconds` seconds
* **Sample Size:** `1000`
* **Average Time per Experiment:** `0.00643` seconds

---

## 2. Profiling Summary (Bottlenecks)
Based on `cProfile` and manual timing hooks, the following table summarizes where the simulation spends the most time.

| Component | Total Time (s) | Avg Time/Exp (s) | % of Total |
| :--- | :--- | :--- | :--- |
| **AKDE Evaluation** | `5.823431` | `0.647048` | `90.6%` |
| **MKDE Evaluation** | `0.092562` | `0.010285` | `1.4%` |
| **AKDE Construction** | `0.053860` | `0.005984` | `0.8%` |
| **MKDE Construction** | `0.000017` | `0.000002` | `<0.1%` |
| **DDE Construction** | `0.000013` | `0.000001` | `<0.1%` |
| **KDE Construction** | `0.000016` | `0.000002` | `<0.1%` |
| **Data Generation** | `0.000357` | `0.000040` | `<0.1%` |
| **IMSE Calculation** | `0.000102` | `0.000011` | `<0.1%` |


**Key Finding:** The primary bottleneck is `AKDE`, then `MKDE`.

## 3. Computational Complexity Analysis


### Runtime Visualization

![Runtime vs Sample Size](/Proj3/results/figures/runtime_subplots.png)

---
### Empirical Evidence
* **Linear Scaling:** Standard KDE and DDE show strictly linear growth with $N$ in the runtime plots.
* **Non-Linear Scaling:** AKDE shows signs of quadratic growth ($N^2$) during the construction phase because the pilot estimate requires an $O(N)$ operation for each of the $N$ data points.

---

## 4. Stability & Numerical Warnings
This section documents numerical issues captured by `numpy_errors.log`.

**Log File Analysis:** `numpy_errors.log`

| Warning Type | Frequency | Conditions | Impact |
| :--- | :--- | :--- | :--- |
| `RuntimeWarning: underflow encountered in exp` | `[Insert Count]` | Occurs when calculating kernels for data points very far from the evaluation grid (tails). The result is so small it rounds to zero. | Low: Safely treated as zero density; practically no impact on accuracy. |
| `RuntimeWarning: underflow encountered in divide` | `[Insert Count]` | Occurs when dividing by extremely small bandwidths or densities that have underflowed to zero. | Medium: Can lead to instability if not caught, but often results in `inf` which may later be handled or clipped. |

**Mitigation Strategy:**
Current strategy uses `np.seterr(all='log')` to capture these without halting execution. Future iterations should implement bounds checking for bandwidth denominators.
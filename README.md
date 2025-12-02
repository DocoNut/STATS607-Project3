# Multi Kernel Density Estimation (MKDE)

This project introduces a new **density estimator** based on the traditional Kernel Density Estimator (KDE).  
Our MKDE method is **unbiased** and often more powerful than the original KDE.

---

## 📌 Features

### src

-**dgps.py**- – Generate samples and PDFs for F, Normal, Beta, and Bimodal distributions.

-**metrics.py**- – Variance and coefficient calculations for MKDE.

-**methods.py**- – Baseline (sequential) implementations of MKDE, AKDE, DDE, and Standard KDE.

-**methods_opt.py**- – Optimized (vectorized) implementations of density estimators for high-performance computing.

-**simulation.py**- – Main driver script. Compares estimators on simulated datasets. Supports --parallel and --profile_mode flags.

-**runtime_baseline.py**- – Analyzes computational complexity (O(N) vs O(N 
2
 )) of baseline methods.

-**comparison.py**- – Benchmarking script to visualize speedup factors between baseline and optimized versions.

### Tests

-**tests/data_test.py**- – Unit tests for data generation validity.

-**tests/function_test.py**- – Unit tests for core mathematical functions.

-**tests/reproducibility_test.py**- – Ensures random seeds produce deterministic results.

-**src/regression_test.py**- – Verifies that optimized methods return identical numerical results to baseline methods.

---

## ⚙️ Installation

We recommend running the project in a **virtual environment**.

The project requires the following Python packages (see `requirements.txt` for full details):

- numpy  
- scipy  
- matplotlib  
- pandas  
- pytest  
- tqdm

Install them and the virtual enviroment with:

```bash
make install
## 📦 Dependencies
```

## 🚀 Usage

### Core Pipeline
Run the standard simulation (sequential execution):
```bash
make simulate
```

Run the optimized simulation (parallel execution with profiling):
```bash
make parallel
```
Note: This runs the main simulation using the optimized methods and generates profile_summary_para.txt.

### Performance Analysis

Run the profiler on the baseline code:
```bash
make profile
```

Generate runtime complexity plots (Time vs. Sample Size):
```bash
make complexity
```

Run the performance benchmark (Speedup Factor comparison):
```bash
make benchmark
```

### Testing

Run the standard unit tests (functionality, data validity, reproducibility):
```bash
make test
```

Run regression tests (verify optimized methods match baseline output):
```bash
make test_regression
```

### Cleanup

Remove all generated results, logs, and figures:

```bash
make clean
```
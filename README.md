# Multi Kernel Density Estimation (MKDE)

This project introduces a new **density estimator** based on the traditional Kernel Density Estimator (KDE).  
Our MKDE method is **unbiased** and often more powerful than the original KDE.

---

## 📌 Features

### src
- **`dgps.py`** – Generate samples and PDFs for F, Normal, Beta, and Bimodal distributions.  
- **`metrics.py`** – variance and coefficients for MKDE .  
- **`methods.py`** – MKDE and other density estimators
- **`simulation.py`** – Compare MKDE with other density estimators on the *Old Faithful* dataset.  

### Tests
- **`tests/data_test.py`** – Unit tests for data.  
- **`tests/function_test.py`** – Unit tests for core model functions.  
- **`tests/reproducibility_test.py`** – Test whether MKDE give the same result under the same random seed.

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


Run the full pipeline (data preprocessing, comparison, and real data test):

```bash
make all
```
or
```bash
make simulate
```

Run tests:
```bash
make test
```

clean the results:
```bash
make clean
```


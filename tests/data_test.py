import pandas as pd
import numpy as np

data = pd.read_csv("data/simulated.csv")

# 1. Non-empty
if data.empty:
    raise ValueError("Data is empty")

# 2. No missing values
if data.isnull().values.any():
    raise ValueError("Data contains missing values")

# 3. All numeric
if not np.all([np.issubdtype(dtype, np.number) for dtype in data.dtypes]):
    raise TypeError("Data contains non-numeric values")

# 4. Shape check (example: expect 2 columns)
if data.shape[1] != 1:
    raise ValueError(f"Expected 2 columns, got {data.shape[1]}")

print("✅ Data tests all passed!")


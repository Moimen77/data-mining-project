
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Load the dataset
data_path = Path(__file__).parent / "data" / "train.csv"
df = pd.read_csv(data_path)

# 2. Display the first 5 rows
print("\n--- First 5 Rows ---")
print(df.head())

# 3. Dataset dimensions
print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# 4. Column names and data types
print("\n--- Dataset Information ---")
df.info()

# 5. Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe(include="all").T)

# 6. Missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 7. Duplicate rows
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# 8. Target distribution
print("\n--- Survival Distribution ---")
print(df["Survived"].value_counts())
print(df["Survived"].value_counts(normalize=True) * 100)

print("\nEDA inspection completed.")

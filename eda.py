
# ==========================================
# Titanic Data Mining Project
# Phase 1: Exploratory Data Analysis (EDA)
# ==========================================

# 1. Imports & Configuration
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


# 2. Load Dataset
def load_data():
    """Load the original Titanic dataset."""
    data_path = Path(__file__).parent / "data" / "train.csv"
    df = pd.read_csv(data_path)
    return df


# 3. Basic Inspection
def inspect_data(df):
    """Display basic information about the dataset."""

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())

    print("\n========== DATA TYPES ==========")
    df.info()

    print("\n========== STATISTICAL SUMMARY ==========")
    print(df.describe(include="all").T)

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATE ROWS ==========")
    print(df.duplicated().sum())


# 4. Target Variable Analysis
def analyze_target(df):
    """Analyze the target variable: Survived."""

    print("\n========== SURVIVAL DISTRIBUTION ==========")
    counts = df["Survived"].value_counts().sort_index()
    percentages = df["Survived"].value_counts(
        normalize=True
    ).sort_index() * 100

    print(counts)
    print("\nPercentages:")
    print(percentages.round(2))


# 5. Main Function
def main():
    df = load_data()

    inspect_data(df)
    analyze_target(df)


if __name__ == "__main__":
    main()

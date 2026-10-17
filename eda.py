
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

    summary = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2)
    })
    print(summary)


# 5. Plot Survival Distribution
def plot_survival_distribution(df):
    """Plot the number of passengers who survived and did not survive."""

    plt.figure(figsize=(7, 5))

    sns.countplot(data=df, x="Survived")

    plt.title("Survival Distribution")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Number of Passengers")

    plt.tight_layout()
    plt.show()


# 6. Plot Survival by Gender
def plot_survival_by_gender(df):
    """Compare survival counts by gender."""

    plt.figure(figsize=(7, 5))

    sns.countplot(data=df, x="Sex", hue="Survived")

    plt.title("Survival by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Number of Passengers")

    plt.legend(
        title="Survived",
        labels=["No", "Yes"]
    )

    plt.tight_layout()
    plt.show()



# 5. Main Function
def main():
    df = load_data()

   # inspect_data(df)
   # analyze_target(df)
   # plot_survival_distribution(df)
    plot_survival_by_gender(df)


if __name__ == "__main__":
    main()

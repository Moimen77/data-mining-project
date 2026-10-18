# Titanic Survival Prediction - Data Mining Project

This repository contains an end-to-end Machine Learning and Data Mining pipeline for predicting passenger survival on the Titanic dataset.

---

## 1. Project Overview & Workflow

The workflow consists of four modular phases:
1. **Exploratory Data Analysis (`eda.py`)**: Data profiling, distribution visualization, demographic breakdowns.
2. **Data Preprocessing (`preprocessing.py`)**: Feature selection, missing value imputation, one-hot encoding, and feature scaling without data leakage. Saves processed datasets and fitted preprocessor.
3. **Model Training & Evaluation (`train_models.py`)**: Trains and compares 4 classification models using unified holdout validation, computes confusion matrices and classification reports, creates metric comparison plots, and serializes the best model.
4. **Final Inference & Submission (`predict.py`)**: Preprocesses the unlabelled Titanic test dataset (`data/test.csv`) using the persisted pipeline and generates the final predictions file.

---

## 2. Methodology & Leakage Prevention

### Train / Validation Split
- Source: `data/train.csv` (891 rows)
- Split: Stratified 80/20 train/validation split (`test_size=0.20`, `random_state=42`, `stratify=y`).
- Rationale: Stratification guarantees the class balance (38.38% survived) is identically preserved across both the 712 training samples and 179 validation samples.

### Preprocessing & Data Leakage Safeguards
- Preprocessing is fitted **strictly on the 80% training split** (`preprocessor.fit_transform(X_train)`).
- The 20% validation split and unlabelled test dataset are transformed using the pre-fitted pipeline (`preprocessor.transform(...)`).
- Imputation (median for numerical, mode for categorical) and scaling statistics (mean and standard deviation) are strictly derived from the training set, eliminating data leakage.

### Features Used
- **Numerical (5)**: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare` (Imputed with median, scaled with `StandardScaler`).
- **Categorical (2)**: `Sex`, `Embarked` (Imputed with mode, encoded with `OneHotEncoder(handle_unknown='ignore')`).
- **Dropped Columns**: `PassengerId`, `Name`, `Ticket`, `Cabin` (high cardinality or metadata identifiers).

---

## 3. Evaluated Models

Four supervised classification algorithms are trained and evaluated on the exact same holdout split:
1. **Decision Tree** (`DecisionTreeClassifier(random_state=42)`)
2. **Random Forest** (`RandomForestClassifier(n_estimators=100, random_state=42)`)
3. **Logistic Regression** (`LogisticRegression(max_iter=1000, random_state=42)`)
4. **Naive Bayes** (`GaussianNB()`)

---

## 4. Evaluation Results

### Performance Summary (`model_comparison.csv`)

| Model | Accuracy | Precision | Recall | F1-score |
|---|:---:|:---:|:---:|:---:|
| **Decision Tree** | **0.8156** | **0.7903** | **0.7101** | **0.7481** |
| **Random Forest** | 0.8156 | 0.8000 | 0.6957 | 0.7442 |
| **Logistic Regression** | 0.8045 | 0.7931 | 0.6667 | 0.7244 |
| **Naive Bayes** | 0.7877 | 0.7385 | 0.6957 | 0.7164 |

*Sorted by F1-score descending. Decision Tree achieved the highest F1-score (0.7481) and tied for highest Accuracy (0.8156).*

### Confusion Matrices (Validation Set: N = 179)

- **Decision Tree**:
  - TN: 97, FP: 13, FN: 20, TP: 49
- **Random Forest**:
  - TN: 98, FP: 12, FN: 21, TP: 48
- **Logistic Regression**:
  - TN: 98, FP: 12, FN: 23, TP: 46
- **Naive Bayes**:
  - TN: 93, FP: 17, FN: 21, TP: 48

---

## 5. Visualizations

- `figures/model_comparison.png` (also at root: `model_comparison.png`): Grouped bar chart comparing Accuracy, Precision, Recall, and F1-score across all 4 models with metric value annotations.
- `figures/confusion_matrices.png`: 2x2 heatmap grid illustrating True/False Positives and Negatives for each model.

---

## 6. Project Directory Structure

```text
titanic project/
│
├── data/
│   ├── train.csv                      # Original labelled dataset (891 rows)
│   ├── test.csv                       # Original unlabelled test dataset (418 rows)
│   └── gender_submission.csv          # Sample submission format
│
├── processed_data/
│   ├── train_processed.csv            # 80% training feature matrix + Survived (712 rows)
│   ├── test_processed.csv             # 20% validation feature matrix + Survived (179 rows)
│   └── preprocessor.joblib            # Serialized ColumnTransformer preprocessing pipeline
│
├── models/
│   └── best_model.joblib              # Serialized best model (Decision Tree)
│
├── figures/
│   ├── model_comparison.png           # Visual performance comparison bar chart
│   └── confusion_matrices.png         # Heatmap of confusion matrices
│
├── eda.py                             # Exploratory Data Analysis script
├── preprocessing.py                   # Data preprocessing and pipeline fitting
├── train_models.py                    # Model training, evaluation, comparison, and export
├── predict.py                         # Test set inference and prediction generation
├── model_comparison.csv               # Exported metrics comparison table
├── model_comparison.png               # Root copy of metric comparison chart
├── predictions.csv                    # Final predictions for test.csv
├── submission.csv                     # Submission file in Kaggle format
└── README.md                          # Project documentation
```

---

## 7. How to Run the Code

Run the scripts in sequence using Python 3.10+:

```bash
# 1. (Optional) Run Exploratory Data Analysis
python eda.py

# 2. Run Data Preprocessing (generates processed datasets and saves preprocessor)
python preprocessing.py

# 3. Train and Evaluate Models (prints metrics, generates plots, saves best model)
python train_models.py

# 4. Generate Final Predictions for data/test.csv
python predict.py
```

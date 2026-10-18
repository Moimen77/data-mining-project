
from pathlib import Path

import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ==========================================
# 1. Load Processed Data
# ==========================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "processed_data"

train_df = pd.read_csv(DATA_DIR / "train_processed.csv")
test_df = pd.read_csv(DATA_DIR / "test_processed.csv")

print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

# ==========================================
# 2. Separate Features and Target
# ==========================================

target = "Survived"

X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_test = test_df.drop(columns=[target])
y_test = test_df[target]

# Ensure the test features match the training features
X_test = X_test.reindex(columns=X_train.columns)

print("Number of training features:", X_train.shape[1])

# ==========================================
# 3. Define the Models
# ==========================================

models = {
    "Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
}

# ==========================================
# 4. Train and Evaluate Models
# ==========================================

results = []

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Predict survival on unseen test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
    })

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

# ==========================================
# 5. Compare Models
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-score",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# ==========================================
# 6. Save Results
# ==========================================

output_path = BASE_DIR / "model_comparison.csv"

results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")

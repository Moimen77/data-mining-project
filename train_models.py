
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
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
trained_models = {}
confusion_matrices = {}

for model_name, model in models.items():

    print(f"\n{'='*20} {model_name} {'='*20}")

    # Train the model
    model.fit(X_train, y_train)
    trained_models[model_name] = model

    # Predict survival on unseen validation/test data
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

    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[model_name] = cm

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

    print("\nConfusion Matrix:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

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

# ==========================================
# 7. Save Best Model
# ==========================================
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)
best_model_path = models_dir / "best_model.joblib"
joblib.dump(best_model, best_model_path)
print(f"Best model ({best_model_name}) saved to: {best_model_path}")

# ==========================================
# 8. Visual Comparison of Metrics & Confusion Matrices
# ==========================================
figures_dir = BASE_DIR / "figures"
figures_dir.mkdir(exist_ok=True)

# 8.1 Grouped Bar Chart of Model Metrics
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
melted_df = results_df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")
palette = sns.color_palette("muted", len(metrics))

ax = sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette=palette)
plt.title("Model Performance Comparison (Titanic Dataset)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Model", fontsize=12, fontweight="bold")
plt.ylabel("Score", fontsize=12, fontweight="bold")
plt.ylim(0, 1.05)
plt.legend(title="Metric", loc="lower right")

# Add value labels on bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f"{height:.4f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )

plt.tight_layout()
chart_path = BASE_DIR / "model_comparison.png"
figures_chart_path = figures_dir / "model_comparison.png"
plt.savefig(chart_path, dpi=300)
plt.savefig(figures_chart_path, dpi=300)
plt.close()
print(f"Metric comparison figure saved to: {chart_path} and {figures_chart_path}")

# 8.2 Confusion Matrices Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
    ax = axes[idx]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
        xticklabels=["Died (0)", "Survived (1)"],
        yticklabels=["Died (0)", "Survived (1)"],
    )
    ax.set_title(f"{model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.suptitle("Confusion Matrices - Model Evaluation", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
cm_chart_path = figures_dir / "confusion_matrices.png"
plt.savefig(cm_chart_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Confusion matrices figure saved to: {cm_chart_path}")

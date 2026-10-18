from pathlib import Path
import pandas as pd
import joblib

# ==========================================
# 1. Load Original Unlabelled Test Data & Saved Artifacts
# ==========================================

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "test.csv"
PREPROCESSOR_PATH = BASE_DIR / "processed_data" / "preprocessor.joblib"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Test data file not found at: {DATA_PATH}")

if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(
        f"Fitted preprocessor not found at: {PREPROCESSOR_PATH}. "
        "Please run preprocessing.py first."
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}. "
        "Please run train_models.py first."
    )

test_raw = pd.read_csv(DATA_PATH)
print("Original test dataset shape:", test_raw.shape)

# Keep PassengerId for final submission/prediction output
passenger_ids = test_raw["PassengerId"]

# ==========================================
# 2. Prepare Features for Preprocessing
# ==========================================

columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
X_test = test_raw.drop(
    columns=[col for col in columns_to_drop if col in test_raw.columns]
)

# Load fitted preprocessor pipeline and transform test features
preprocessor = joblib.load(PREPROCESSOR_PATH)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
X_test_processed_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)

print("Processed test features shape:", X_test_processed_df.shape)

# ==========================================
# 3. Predict Using Best Model
# ==========================================

best_model = joblib.load(MODEL_PATH)
print("Loaded model type:", type(best_model).__name__)

predictions = best_model.predict(X_test_processed_df)

# ==========================================
# 4. Save Final Predictions File
# ==========================================

submission_df = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": predictions
})

output_file = BASE_DIR / "predictions.csv"
submission_df.to_csv(output_file, index=False)
print(f"\nFinal predictions saved successfully to: {output_file}")

# Also save submission.csv (standard Kaggle format)
submission_file = BASE_DIR / "submission.csv"
submission_df.to_csv(submission_file, index=False)
print(f"Kaggle submission format saved to: {submission_file}")

# ==========================================
# 5. Prediction Summary
# ==========================================

survived_count = (predictions == 1).sum()
died_count = (predictions == 0).sum()
total = len(predictions)

print("\n========== TEST PREDICTIONS SUMMARY ==========")
print(f"Total test passengers: {total}")
print(f"Predicted Survived (1): {survived_count} ({survived_count / total * 100:.2f}%)")
print(f"Predicted Died (0):     {died_count} ({died_count / total * 100:.2f}%)")
print("\nFirst 10 predictions:")
print(submission_df.head(10).to_string(index=False))

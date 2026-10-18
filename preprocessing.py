
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 1. Load the dataset
# ==========================================
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "train.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# ==========================================
# 2. Remove duplicate rows
# ==========================================
duplicates = df.duplicated().sum()
print("Duplicate rows found:", duplicates)

df = df.drop_duplicates().copy()

# ==========================================
# 3. Separate features and target
# ==========================================
target = "Survived"

if target not in df.columns:
    raise ValueError("The dataset must contain a 'Survived' column.")

X = df.drop(columns=[target])
y = df[target]

# Remove rows with a missing target, if any
valid_target = y.notna()
X = X.loc[valid_target].copy()
y = y.loc[valid_target].copy()

# ==========================================
# 4. Drop columns not used in this baseline
# ==========================================
columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]

X = X.drop(
    columns=[col for col in columns_to_drop if col in X.columns]
)

# ==========================================
# 5. Identify numerical and categorical columns
# ==========================================
numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:", numeric_features)
print("Categorical features:", categorical_features)

# ==========================================
# 6. Preprocessing pipelines
# ==========================================
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ==========================================
# 7. Split the data BEFORE fitting preprocessing
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# ==========================================
# 8. Fit on training data and transform both sets
# ==========================================
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get the transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Convert processed data to DataFrames
X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)

# ==========================================
# 9. Save the processed datasets
# ==========================================
output_dir = BASE_DIR / "processed_data"
output_dir.mkdir(exist_ok=True)

train_processed = X_train_processed.copy()
train_processed[target] = y_train

test_processed = X_test_processed.copy()
test_processed[target] = y_test

train_processed.to_csv(
    output_dir / "train_processed.csv", index=False
)

test_processed.to_csv(
    output_dir / "test_processed.csv", index=False
)

print("\nPreprocessing completed successfully!")
print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)
print("Missing values in training:", X_train_processed.isnull().sum().sum())
print("Missing values in testing:", X_test_processed.isnull().sum().sum())
print("\nSaved files in:", output_dir)
print("- train_processed.csv")
print("- test_processed.csv")
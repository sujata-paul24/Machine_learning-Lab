import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 1. Create synthetic dataset
data = {
    "Age": [22, 25, None, 30, 28],
    "Salary": [25000, 35000, 30000, None, 45000],
    "Department": ["IT", "HR", "IT", "Finance", "HR"],
    "Years_Experience": [1, 3, 2, None, 5],
    "Purchased": [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

print("--- Original Dataset ---")
print(df)

# Separate features and target
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

numeric_features = ["Age", "Salary", "Years_Experience"]
categorical_features = ["Department"]

# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine transformations
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print("\n--- Processed Feature Matrix ---")
print(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])
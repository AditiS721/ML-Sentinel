from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# ======================================================
# CONFIGURATION
# ======================================================

N_SAMPLES = 100000
N_FEATURES = 12
RANDOM_STATE = 42

# ======================================================
# CREATE DATASET
# ======================================================

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,

    # Important Features
    n_informative=8,

    # Redundant features (linear combinations)
    n_redundant=2,

    # Duplicate features
    n_repeated=0,

    # Binary Classification
    n_classes=2,

    # 70% class 0
    # 30% class 1
    weights=[0.7, 0.3],

    # 1% wrong labels
    flip_y=0.01,

    # Better class separation
    class_sep=1.2,

    # Shuffle features
    shuffle=True,

    random_state=RANDOM_STATE
)

# ======================================================
# FEATURE NAMES
# ======================================================

feature_names = [
    "age_score",
    "income_score",
    "credit_score",
    "loan_amount",
    "transaction_freq",
    "device_score",
    "location_score",
    "account_age",
    "purchase_score",
    "behavior_score",
    "risk_score",
    "activity_score"
]

df = pd.DataFrame(X, columns=feature_names)

df["target"] = y

# ======================================================
# TRAIN / VALIDATION / TEST SPLIT
# ======================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["target"]
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["target"]
)

# ======================================================
# CREATE DATA DIRECTORY
# ======================================================

os.makedirs("data", exist_ok=True)

# ======================================================
# SAVE FILES
# ======================================================

train_df.to_csv("data/train.csv", index=False)
validation_df.to_csv("data/validation.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

# Save complete dataset

df.to_csv("data/full_dataset.csv", index=False)

# Save reference data
# Drift monitor compares production requests with this

train_df.drop(columns=["target"]).to_csv(
    "data/reference.csv",
    index=False
)

# ======================================================
# PRINT INFORMATION
# ======================================================

print("=" * 60)
print("DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print()

print("Total Dataset Shape:")
print(df.shape)

print()

print("Train Shape:")
print(train_df.shape)

print()

print("Validation Shape:")
print(validation_df.shape)

print()

print("Test Shape:")
print(test_df.shape)

print()

print("Target Distribution:")
print(df["target"].value_counts(normalize=True))

print()

print("Files Generated:")
print("----------------------------")
print("data/full_dataset.csv")
print("data/train.csv")
print("data/validation.csv")
print("data/test.csv")
print("data/reference.csv")

print()

print("First Five Rows:")
print(df.head())
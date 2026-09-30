import os
import joblib
import sklearn
import mlflow
import mlflow.sklearn

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_NAME = "sentinel_classifier"

RANDOM_STATE = 42

N_ESTIMATORS = 200

MAX_DEPTH = 10


# ==========================================================
# DIRECTORIES
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# ==========================================================
# LOAD DATA
# ==========================================================

train_df = pd.read_csv(
    "data/train.csv"
)

val_df = pd.read_csv(
    "data/validation.csv"
)


X_train = train_df.drop(
    columns=["target"]
)

y_train = train_df["target"]


X_val = val_df.drop(
    columns=["target"]
)

y_val = val_df["target"]



# ==========================================================
# MLFLOW CONFIGURATION
# ==========================================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "ML Sentinel"
)


# ==========================================================
# TRAINING
# ==========================================================

with mlflow.start_run():


    # ------------------------------
    # Tags
    # ------------------------------

    mlflow.set_tag(
        "project",
        "ML Sentinel"
    )

    mlflow.set_tag(
        "model_type",
        "Random Forest"
    )


    # ------------------------------
    # Model
    # ------------------------------

    model = RandomForestClassifier(

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


    model.fit(
        X_train,
        y_train
    )


    # ------------------------------
    # Prediction
    # ------------------------------

    predictions = model.predict(
        X_val
    )

    probabilities = model.predict_proba(
        X_val
    )[:,1]


    # ------------------------------
    # Metrics
    # ------------------------------

    accuracy = accuracy_score(
        y_val,
        predictions
    )


    precision = precision_score(
        y_val,
        predictions
    )


    recall = recall_score(
        y_val,
        predictions
    )


    f1 = f1_score(
        y_val,
        predictions
    )


    roc_auc = roc_auc_score(
        y_val,
        probabilities
    )


    print("\nValidation Metrics")
    print("------------------")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC AUC  : {roc_auc:.4f}"
    )


    # ======================================================
    # SAVE MODEL
    # ======================================================

    joblib.dump(
        model,
        "models/model.pkl"
    )



    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    cm = confusion_matrix(
        y_val,
        predictions
    )


    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title(
        "Confusion Matrix"
    )

    plt.savefig(
        "plots/confusion_matrix.png"
    )

    plt.close()



    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================

    importance = pd.DataFrame({

        "feature":
        X_train.columns,

        "importance":
        model.feature_importances_

    })


    importance = importance.sort_values(
        by="importance",
        ascending=False
    )


    importance.to_csv(
        "plots/feature_importance.csv",
        index=False
    )


    plt.figure(
        figsize=(8,6)
    )

    plt.barh(

        importance["feature"],

        importance["importance"]

    )


    plt.title(
        "Feature Importance"
    )


    plt.tight_layout()


    plt.savefig(
        "plots/feature_importance.png"
    )


    plt.close()



    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================


    report = classification_report(
        y_val,
        predictions
    )


    with open(
        "plots/classification_report.txt",
        "w"
    ) as f:

        f.write(report)



    # ======================================================
    # LOG PARAMETERS
    # ======================================================

    mlflow.log_param(
        "model",
        "RandomForest"
    )


    mlflow.log_param(
        "n_estimators",
        N_ESTIMATORS
    )


    mlflow.log_param(
        "max_depth",
        MAX_DEPTH
    )


    mlflow.log_param(
        "random_state",
        RANDOM_STATE
    )


    mlflow.log_param(
        "train_samples",
        len(X_train)
    )


    mlflow.log_param(
        "features",
        X_train.shape[1]
    )


    mlflow.log_param(
        "sklearn_version",
        sklearn.__version__
    )



    # ======================================================
    # LOG METRICS
    # ======================================================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )



    # ======================================================
    # LOG ARTIFACTS
    # ======================================================


    mlflow.log_artifact(
        "plots/confusion_matrix.png"
    )


    mlflow.log_artifact(
        "plots/feature_importance.png"
    )


    mlflow.log_artifact(
        "plots/feature_importance.csv"
    )


    mlflow.log_artifact(
        "plots/classification_report.txt"
    )



    # ======================================================
    # REGISTER MODEL
    # ======================================================


    mlflow.sklearn.log_model(

        sk_model=model,

        name="model",

        registered_model_name=MODEL_NAME

    )



print("\nTraining Completed Successfully!")
print("Model saved -> models/model.pkl")
print("Artifacts saved -> plots/")
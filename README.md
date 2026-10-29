# ML Sentinel

A Python project for serving a binary classification model and monitoring its predictions and input data drift.

## Project Objective

Build a practical ML monitoring workflow: generate synthetic data, train and track a model, serve predictions through an API, simulate incoming requests, and inspect prediction patterns and data drift.

## Implemented Features

- **Dataset generation:** 100,000 synthetic samples with 12 features and stratified train/validation/test splits of 70%/15%/15%.
- **Model training:** Random Forest classifier with validation metrics, a confusion matrix, and feature-importance reports.
- **Experiment tracking:** MLflow logging for parameters, metrics, artifacts, and model registration.
- **Prediction API:** FastAPI endpoint (`POST /predict`) returning a predicted class and class-1 probability.
- **Traffic simulation:** Random requests every two seconds, with successful predictions saved to `logs/predictions.csv`.
- **Dashboard:** Streamlit charts and summaries of logged predictions.
- **Drift reporting:** Evidently compares training features with logged inputs and exports `drift_report.html`.

## Learning Outcomes

- Generate synthetic classification data and create stratified dataset splits.
- Train and evaluate a Random Forest using accuracy, precision, recall, F1, and ROC AUC.
- Track experiments and register models with MLflow.
- Serve model predictions through a validated FastAPI interface.
- Log simulated inference traffic and visualize prediction patterns.
- Detect input distribution changes using data drift reports.

## Tech Stack

Python, scikit-learn, pandas, MLflow, FastAPI, Streamlit, Plotly, and Evidently.

## Current Scope

This is a synthetic-data demonstration; feature names do not represent validated financial risk signals. Drift reporting is run manually, and the dashboard reads the prediction log when executed or rerun. Automated retraining and chaos testing are not implemented; their directories currently contain dependency files only.

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "framingham_stage2_engineered.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "heart_disease_logistic_pipeline.pkl"
)


# --------------------------------------------------
# Load data and model
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Prepare data
# --------------------------------------------------

target_col = "TenYearCHD"

redundant_flag_cols = [
    "High_BP_Flag",
    "High_Chol_Flag",
    "Smoker_Flag",
    "Diabetes_Flag",
    "High_BMI_Flag"
]

X = df.drop(
    columns=[target_col] + redundant_flag_cols
)

y = df[target_col]


# --------------------------------------------------
# Same train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()


# Sensitivity = Recall
sensitivity = tp / (tp + fn)

# Specificity
specificity = tn / (tn + fp)


# --------------------------------------------------
# Save results
# --------------------------------------------------

results = {
    "Accuracy": round(accuracy, 4),
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1 Score": round(f1, 4),
    "ROC-AUC": round(roc_auc, 4),
    "Sensitivity": round(sensitivity, 4),
    "Specificity": round(specificity, 4)
}


results_df = pd.DataFrame(
    [results]
)


OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "evaluation_results.csv"
)


results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nModel Evaluation Results")
print("------------------------")

for metric, value in results.items():

    print(
        f"{metric}: {value}"
    )


print("\nConfusion Matrix")
print("----------------")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print(
    f"\nTN: {tn}"
)

print(
    f"FP: {fp}"
)

print(
    f"FN: {fn}"
)

print(
    f"TP: {tp}"
)

print(
    f"\nEvaluation saved to:"
)

print(
    OUTPUT_PATH
)
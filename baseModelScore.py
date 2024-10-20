from sklearn.preprocessing import*
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,precision_score,recall_score,f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def evaluate_models(df):

    #creating dependent and independent features
    X = df.drop("target",axis=1)
    y = df["target"]

    # standardizing the data 
    scaler = StandardScaler()
    X =scaler.fit_transform(X)

    # trian test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier(random_state=24),
        "XGBoost": XGBClassifier(random_state=24),
        "LightGBM": LGBMClassifier(verbosity = -1)
    }

    # Initialize a list to store metrics
    metrics_list = []

    # Iterate through models
    for model_name, model in models.items():
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Collect metrics for training
        metrics_train = {
            'Model': model_name,
            'Accuracy_Train': accuracy_score(y_train, y_train_pred),
            'Precision_Train': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'Recall_Train': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'F1_Score_Train': f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        }

        # Check if the model supports ROC AUC score calculation
        if hasattr(model, "predict_proba"):
            metrics_train['AUC_ROC_Train'] = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        else:
            warnings.warn(f"{model_name} does not support probability predictions for ROC AUC.")

        # Collect metrics for testing
        metrics_test = {
            'Accuracy_Test': accuracy_score(y_test, y_test_pred),
            'Precision_Test': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
            'Recall_Test': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
            'F1_Score_Test': f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        }

        # Check if the model supports ROC AUC score calculation
        if hasattr(model, "predict_proba"):
            metrics_test['AUC_ROC_Test'] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        else:
            warnings.warn(f"{model_name} does not support probability predictions for ROC AUC.")

        # Combine train and test metrics into one dictionary
        combined_metrics = {**metrics_train, **metrics_test}

        # Append to the list
        metrics_list.append(combined_metrics)

    # Create a DataFrame from the list of metrics
    metrics_df = pd.DataFrame(metrics_list)

    return metrics_df

# Example usage:
# Assuming X_train, y_train, X_test, and y_test are already defined.
# metrics_df = evaluate_models(X_train, y_train, X_test, y_test)
# print(metrics_df)
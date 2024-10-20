import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import optuna
import warnings
from sklearn.preprocessing import StandardScaler

def hyperparam_models(data, target_column):
    # Split the DataFrame into features and target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models and their hyperparameters for GridSearchCV
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(),
            "params": {
                'C': [0.001, 0.01, 0.1,0.5,1,5,10],
                'solver': ['lbfgs', 'liblinear','newton-cg', 'newton-cholesky']
            },
            "use_optuna": False
        },
        "RandomForestClassifier": {
            "model": RandomForestClassifier(random_state=24),
            "params": {
                "max_depth":[1,2,3,4],
                "min_samples_split":[2,4],
                "min_samples_leaf":[2,4],
            },
            "use_optuna": False
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=24),
            "params": None,
            "use_optuna": True
        },
        "LightGBM": {
            "model": LGBMClassifier(verbosity = -1),
            "params": None,
            "use_optuna": True
        }
    }

    # Initialize lists to store metrics for GridSearchCV and Optuna
    gridsearch_metrics_list = []
    optuna_metrics_list = []

    # Iterate through models
    for model_name, model_info in models.items():
        model = model_info["model"]
        
        if model_info["use_optuna"]:
            # Define the objective function for Optuna
            def objective(trial):
                if model_name == "XGBoost":
                    param = {
                        'max_depth': trial.suggest_int('max_depth', 1, 10),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)

                    }
                    model.set_params(**param)
                elif model_name == "LightGBM":
                    param = {
                        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200)
                    }
                    model.set_params(**param)

                # Fit the model and return accuracy as the objective value
                model.fit(X_train, y_train)
                return accuracy_score(y_test, model.predict(X_test))

            # Create an Optuna study and optimize the objective function
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)

            # Get best parameters from Optuna study
            best_params = study.best_params
            model.set_params(**best_params)

            # Fit the final model with best parameters on training data
            model.fit(X_train, y_train)

            # Predictions on train and test data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Collect metrics for training and testing for Optuna
            metrics = {
                'Model': model_name,
                'Accuracy_Train': accuracy_score(y_train, y_train_pred),
                'Precision_Train': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'Recall_Train': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'F1_Score_Train': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'Accuracy_Test': accuracy_score(y_test, y_test_pred),
                'Precision_Test': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'Recall_Test': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'F1_Score_Test': f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            }

            # Append metrics to the Optuna list
            optuna_metrics_list.append(metrics)

        else:
            # Use GridSearchCV for Logistic Regression and Random Forest
            grid_search = GridSearchCV(estimator=model,
                                       param_grid=model_info["params"],
                                       scoring='accuracy',
                                       cv=5,
                                       n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            model.set_params(**best_params)

            # Fit the final model with best parameters on training data
            model.fit(X_train, y_train)

            # Predictions on train and test data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Collect metrics for training and testing for GridSearchCV
            metrics = {
                'Model': model_name,
                'Accuracy_Train': accuracy_score(y_train, y_train_pred),
                'Precision_Train': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'Recall_Train': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'F1_Score_Train': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'Accuracy_Test': accuracy_score(y_test, y_test_pred),
                'Precision_Test': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'Recall_Test': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'F1_Score_Test': f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            }

            # Append metrics to the GridSearchCV list
            gridsearch_metrics_list.append(metrics)

    # Create DataFrames from the lists of metrics
    gridsearch_metrics_df = pd.DataFrame(gridsearch_metrics_list)
    optuna_metrics_df = pd.DataFrame(optuna_metrics_list)

    return gridsearch_metrics_df, optuna_metrics_df

# Example usage:
# Assuming data is a pandas DataFrame with a target column named 'target'.
# metrics_gridsearch_df, metrics_optuna_df = evaluate_models(data=data_frame_with_target_column,'target')
# print("GridSearchCV Results:\n", metrics_gridsearch_df)
# print("Optuna Results:\n", metrics_optuna_df)
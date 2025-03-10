import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import statsmodels.api as sm
import subprocess 
import psutil
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import mlflow
from elasticsearch import Elasticsearch
import json
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC

from joblib import dump
from sklearn.model_selection import GridSearchCV


import argparse
import os
import pandas as pd
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

import seaborn as sns
import psutil  # For system metrics

import json
from datetime import datetime
from elasticsearch import Elasticsearch

mlflow.enable_system_metrics_logging()
from datetime import datetime

es = Elasticsearch([{"host": "localhost", "port": 9200, "scheme": "http"}])

def prepare_data(train_path, test_path):

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    col_num = train_data.select_dtypes(include=["int64", "float64"]).columns

    train_data = cap_outliers(train_data, col_num)
    test_data = cap_outliers(test_data, col_num)

    train_data, label_encoders = encode_categorical_features(train_data)
    test_data, _ = encode_categorical_features(test_data)

    columns_to_drop = [
        "Total day charge",
        "Total eve charge",
        "Total night charge",
        "Total intl charge",
        "Voice mail plan",
    ]
    train_data = train_data.drop(columns=columns_to_drop)
    test_data = test_data.drop(columns=columns_to_drop)

    X_train = train_data.drop(columns=["Churn"])
    y_train = train_data["Churn"]
    X_test = test_data.drop(columns=["Churn"])
    y_test = test_data["Churn"]

    x_log = sm.add_constant(X_train)
    reg_log = sm.Logit(y_train, x_log)
    results_log = reg_log.fit()
    significant_features = results_log.pvalues[results_log.pvalues < 0.05].index
    X_train = X_train[significant_features.drop("const")]
    X_test = X_test[significant_features.drop("const")]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_st = scaler.transform(X_train)
    X_test_st = scaler.transform(X_test)
    pd.DataFrame(X_train_st, columns=X_train.columns).to_csv("X_train.csv", index=False)
    pd.DataFrame(X_test_st, columns=X_test.columns).to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    joblib.dump(scaler, "scaler.pkl")
    return scaler, label_encoders


def cap_outliers(data, col_num):
    for col_ in col_num:
        Q1 = data[col_].quantile(0.25)
        Q3 = data[col_].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col_] = data[col_].clip(lower=lower_bound, upper=upper_bound)
    return data


def encode_categorical_features(data):
    encoded_data = data.copy()
    label_encoders = {}
    binary_features = ["International plan", "Voice mail plan", "Churn"]
    for feature in binary_features:
        le = LabelEncoder()
        encoded_data[feature] = le.fit_transform(encoded_data[feature])
        label_encoders[feature] = le
    le_state = LabelEncoder()
    encoded_data["State"] = le_state.fit_transform(encoded_data["State"])
    label_encoders["State"] = le_state
    return encoded_data, label_encoders


def train_model(X_train, y_train):

    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    mlflow.sklearn.log_model(svm_model, "model")
    return svm_model






# Assuming Elasticsearch is initialized elsewhere


def log_to_elasticsearch(params=None, metrics=None, cm=None):
    """
    Logs the model parameters, metrics, and confusion matrix to Elasticsearch.

    Args:
        params (dict, optional): Dictionary containing model parameters. Defaults to None.
        metrics (dict, optional): Dictionary containing evaluation metrics. Defaults to None.
        cm (list, optional): Confusion matrix in list format. Defaults to None.
    """
    # Check if any of the inputs is empty
    if (params is None or len(params) == 0) or (metrics is None or len(metrics) == 0) or (cm is None or len(cm) == 0):
        print("No parameters, metrics, or confusion matrix to log.")
        return

    # Prepare the log data
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "params": params,
        "metrics": metrics,
        "confusion_matrix": cm.tolist()  
    }

    # Log to Elasticsearch
    try:
        response = es.index(index="mlflow-logs", body=json.dumps(log_data))
        print("Model parameters, metrics, and confusion matrix sent to Elasticsearch.")
        print("Elasticsearch response:", response)
    except Exception as e:
        print(f"Error sending logs to Elasticsearch: {e}")
    

def evaluate_model(model, X_test, y_test, model_name="model"):
    # Make predictions
    y_pred = model.predict(X_test)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print metrics for debugging
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Log model parameters
    if hasattr(model, "get_params"):
        model_params = model.get_params()
        mlflow.log_params(model_params)
        print("Model parameters logged.")

    # Log the model as an artifact
    mlflow.sklearn.log_model(model, model_name)
    print(f"Model logged as artifact with name: {model_name}")

    # Log system metrics in the System Metrics page
    log_system_metrics_function()

    # Log confusion matrix as an artifact
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")

    cm_image_path = "confusion_matrix.png"
    plt.savefig(cm_image_path)
    mlflow.log_artifact(cm_image_path)
    print(f"Confusion matrix image saved and logged as artifact: {cm_image_path}")

    # Log ROC curve and AUC (if you have this function defined)
    log_roc_auc(model, X_test, y_test)

    # Log evaluation metrics
    try:
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        print("Evaluation metrics logged.")
        
    except Exception as e:
        print(f"Error logging metrics: {e}")
        
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    # Elasticsearch logging
    params = {
        "kernel": model_params.get("kernel") if "kernel" in model_params else None,
        "C": model_params.get("C") if "C" in model_params else None,
        "degree": model_params.get("degree") if "degree" in model_params else None,
        "coef0": model_params.get("coef0") if "coef0" in model_params else None,
        "gamma": model_params.get("gamma") if "gamma" in model_params else None,
        "timestamp": datetime.utcnow().isoformat(),  # Convert datetime to string
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    try:
        # Index into Elasticsearch
        es.index(index="mlflow-logs", body=json.dumps({"params": params}))
        print("Metrics and model parameters sent to Elasticsearch.")
    except Exception as e:
        print(f"Error sending logs to Elasticsearch: {e}")

    return accuracy, precision, recall, f1, cm


def improve_model(X_train, y_train, X_test, y_test):
    print("# Tuning hyper-parameters for F1 Score")

    # Define the hyperparameters to tune
    tuned_parameters = [
        {"kernel": ["linear"], "C": [0.1, 1, 10, 50], "coef0": [0, 0.5, 1]},
        {
            "kernel": ["poly"],
            "C": [0.1, 1, 10, 50],
            "degree": [2, 3, 4, 5, 6],
            "coef0": [0, 0.5, 1],
            "gamma": ["scale", "auto"],
        },
        {
            "kernel": ["rbf"],
            "gamma": [1e-3, 1e-4],
            "C": [0.1, 1, 10, 50],
            "coef0": [0, 0.5, 1],
        },
        {
            "kernel": ["sigmoid"],
            "gamma": ["scale", "auto"],
            "C": [0.1, 1, 10, 50],
            "coef0": [0, 0.5, 1],
        },
    ]

    # GridSearchCV for hyperparameter tuning
    clf = GridSearchCV(SVC(), tuned_parameters, scoring="f1", cv=5)
    clf.fit(X_train, y_train)

    print("Best Parameters:", clf.best_params_)
    print("Best Cross-Validation Score:", clf.best_score_)

    # Best model after GridSearchCV
    best_svm = clf.best_estimator_
    best_params = clf.best_params_

    # Prediction with the best model
    y_pred_best_svm = best_svm.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best_svm)
    print("Confusion Matrix:")
    print(cm)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred_best_svm)
    precision = precision_score(y_test, y_pred_best_svm, zero_division=1)
    recall = recall_score(y_test, y_pred_best_svm, zero_division=1)
    f1 = f1_score(y_test, y_pred_best_svm, zero_division=1)

    # Display metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Log the metrics and confusion matrix to MLflow
    try:
        mlflow.log_params(best_params)  # Log best params from GridSearchCV
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        print("Metrics and model parameters logged to MLflow.")
    except Exception as e:
        print(f"Error logging to MLflow: {e}")

    # Log the confusion matrix as an artifact
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")

        cm_image_path = "confusion_matrix.png"
        plt.savefig(cm_image_path)
        mlflow.log_artifact(cm_image_path)
        print(f"Confusion matrix image saved and logged as artifact: {cm_image_path}")
    except Exception as e:
        print(f"Error logging confusion matrix: {e}")

    # Elasticsearch logging
    params = {
        "kernel": best_params.get("kernel"),
        "C": best_params.get("C"),
        "degree": best_params.get("degree"),
        "coef0": best_params.get("coef0"),
        "gamma": best_params.get("gamma"),
        "timestamp": datetime.utcnow().isoformat(),  # Convert datetime to string
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    try:
        # Index into Elasticsearch
        es.index(index="mlflow-logs", body=json.dumps({"params": params}))
        print("Metrics and model parameters sent to Elasticsearch.")
    except Exception as e:
        print(f"Error sending logs to Elasticsearch: {e}")

    return best_svm, best_params


def save_model(model, filename="best_svm_model.pkl"):

    joblib.dump(model, filename)
    print(f" Model saved in file :: '{filename}'.")


"""
def load_model(filename="best_svm_model.pkl"):

    model = joblib.load(filename)
    print(f" Model loaded from '{filename}'.")
    return model
"""


def load_model(deployment):

    model = None  # Initialize model as None
    training_runs = None
    experiment_name = "Churn_Prediction_Experiment"

    # Get the experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Check if the experiment exists
    if experiment:
        experiment_id = experiment.experiment_id
        print(f"Experiment ID for '{experiment_name}': {experiment_id}")
    else:
        print(f"Experiment '{experiment_name}' does not exist.")

    # Step 1: Search for the most recent run
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id], order_by=["start_time DESC"]
    )

    # Step 2: Filter runs based on deployment type
    if deployment:
        training_runs = runs[runs["tags.mlflow.runName"] == deployment]
    else:
        training_runs = runs[
            runs["tags.mlflow.runName"].isin(["SVM_Training", "SVM_Improvement"])
        ]

    if not training_runs.empty:
        # Get the latest run
        latest_run = training_runs.iloc[0]
        run_id = latest_run["run_id"]
        run_name = latest_run["tags.mlflow.runName"]
        print(f"Found Latest Run ID: {run_id} ({run_name})")

        # List artifacts in the run
        try:
            artifacts = mlflow.artifacts.list_artifacts(run_id)
            print("Artifacts in the run:")
            for artifact in artifacts:
                print(f" - {artifact.path}")
        except Exception as e:
            print(f"Error listing artifacts: {e}")

        # Load the model
        model_uri = f"runs:/{run_id}/model"
        print(f"Model URI: {model_uri}")

        try:
            model = mlflow.sklearn.load_model(model_uri)
            if model:
                print("Model loaded successfully!")
            else:
                print("Failed to load the model.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("No matching run found in the default experiment.")

    return model


mlflow.set_tracking_uri("http://localhost:5001")


def retraine_svm(C, kernel, degree, gamma, coef0, random_state):
    # Set the experiment name
    mlflow.set_experiment("Churn_Prediction_Experiment")

    # Load datasets
    try:
        x_train = pd.read_csv("X_train.csv").values
        x_test = pd.read_csv("X_test.csv").values
        y_train = pd.read_csv("y_train.csv").values
        y_test = pd.read_csv("y_test.csv").values
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None  # Exit function if datasets fail to load

    # Initialize and train the SVM model
    model = SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,  # Add coef0 here
        random_state=random_state,
    )
    model.fit(x_train, y_train)

    # Evaluate the model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics and parameters to MLflow
    with mlflow.start_run(run_name="new trained model", log_system_metrics=True) as run:
        # Log hyperparameters
        run_id = run.info.run_id  # Get the run ID
        print(f"Run ID: {run_id}")
        mlflow.log_param("C", C)
        mlflow.log_param("kernel", kernel)
        mlflow.log_param("degree", degree)
        mlflow.log_param("gamma", gamma)
        mlflow.log_param("coef0", coef0)  # Log coef0
        mlflow.log_param("random_state", random_state)
        # Register the model in the Model Registry
        model_uri = f"runs:/{run_id}/model"
        model_name = "Churn_Prediction_Model"
        mlflow.register_model(model_uri, model_name)
        print(f"Model registered as '{model_name}'.")

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

    # Define the local storage path

    # Save the model locally to the specified path
    dump(model, "svm_model.joblib")

    # Return response
    return accuracy, precision, recall, f1


def log_system_metrics_function():
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    mlflow.log_metric("cpu_usage", cpu_usage)
    mlflow.log_metric("memory_usage", memory_usage)
    print(
        f"Logged system metrics: CPU Usage = {cpu_usage}%, Memory Usage = {memory_usage}%"
    )






# Function to log ROC curve and AUC
def log_roc_auc(model, X_test, y_test):
    # Get predicted probabilities for the positive class
    y_pred_proba = model.decision_function(X_test)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")

    # Save ROC curve plot
    roc_curve_path = "roc_curve.png"
    plt.savefig(roc_curve_path)
    plt.close()

    # Log ROC curve and AUC
    mlflow.log_metric("auc", roc_auc)
    mlflow.log_artifact(roc_curve_path)
    print(f"ROC curve and AUC logged. AUC = {roc_auc:.2f}")

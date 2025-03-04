import argparse
import os
import pandas as pd
import joblib
import numpy as np
import mlflow
#import pynvml
import mlflow.sklearn
import matplotlib.pyplot as plt
import subprocess 
import seaborn as sns
import psutil  # For system metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from model_pipeline import (
    prepare_data,
    train_model,
    improve_model,
    save_model,
    load_model,
    evaluate_model,
    log_system_metrics_function,
    log_roc_auc,
    
    
)
from sklearn.model_selection import GridSearchCV
from datetime import datetime


mlflow.enable_system_metrics_logging()
# Initialize NVML (NVIDIA Management Library)
#pynvml.nvmlInit()

# Get handle for the first GPU (index 0)
#handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Print the name of the GPU
#print(f"GPU Name: {pynvml.nvmlDeviceGetName(handle)}")

# Shutdown NVML
#pynvml.nvmlShutdown()


# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Churn_Prediction_Experiment")

# Define file paths
train_path = "churn-bigml-80.csv"
test_path = "churn-bigml-20.csv"
model_path = "best_svm_model.pkl"


# Define prepared data file paths
X_train_file = "X_train.csv"
X_test_file = "X_test.csv"
y_train_file = "y_train.csv"
y_test_file = "y_test.csv"
requirements_file = "requirements.txt"

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Train, evaluate, and improve an SVM model."
)
parser.add_argument("--prepare", action="store_true", help="Prepare and save data")
parser.add_argument("--train", action="store_true", help="Train the model")
parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
parser.add_argument("--improve", action="store_true", help="Improve the model")
parser.add_argument("--save", action="store_true", help="Save the model")
parser.add_argument(
    "--load", action="store_true", help="Load the saved model and evaluate"
)

args = parser.parse_args()
        
def log_data_files():
    mlflow.log_artifact(X_train_file)
    mlflow.log_artifact(X_test_file)
    mlflow.log_artifact(y_train_file)
    mlflow.log_artifact(y_test_file)
    mlflow.log_artifact(requirements_file)
    print("Data files logged as artifacts.")   





# Data preparation logic
if args.prepare:
    mlflow.end_run()
    with mlflow.start_run(
        run_name="SVM_Preparing_Data", log_system_metrics=True
    ) as run:
        scaler, label_encoders = prepare_data(train_path, test_path)
        print("Data preparation completed and saved.")
        run_id = run.info.run_id  # Get the run ID
        print(f"Run ID: {run_id}")
        # log_system_metrics=True
        # print(mlflow.MlflowClient().get_run(run.info.run_id).data)

        # Log data files
        log_data_files()

        log_system_metrics_function()  # Log system metrics

# Check if prepared data exists before proceeding with other steps
if os.path.exists(X_train_file) and os.path.exists(X_test_file):
    #  saved data
    X_train_st = pd.read_csv(X_train_file).values
    X_test_st = pd.read_csv(X_test_file).values
    y_train = pd.read_csv(y_train_file).values.ravel()
    y_test = pd.read_csv(y_test_file).values.ravel()
else:
    print("Prepared data not found. Run --prepare first.")
    exit()

svm_model = None


# Training logic
if args.train:
    mlflow.end_run()
    with mlflow.start_run(run_name="SVM_Training", log_system_metrics=True) as run:
        print("Training the model...")
        X_train_st = pd.read_csv(X_train_file).values
        y_train = pd.read_csv(y_train_file).values.ravel()

        svm_model = train_model(X_train_st, y_train)
        save_model(svm_model)
        run_id = run.info.run_id  # Get the run ID
        print(f"Run ID: {run_id}")

        # Log model parameters
        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("kernel", "default")  # Default kernel for initial training
        mlflow.log_param("training_data", train_path)
        log_system_metrics_function()  # Log system metrics
        log_data_files()
        # Register the model in the Model Registry
        model_uri = f"runs:/{run_id}/model"
        model_name = "Churn_Prediction_Model"
        mlflow.register_model(model_uri, model_name)
        print(f"Model registered as '{model_name}'.")

        print("Model training complete.")

# Evaluation logic
if args.evaluate:
    svm_model=load_model(deployment=None)
    if svm_model is None:
        print("No trained model available. Run training first.")
    else:
        mlflow.end_run()  # End any existing run
        with mlflow.start_run(
            run_name="SVM_Evaluation", log_system_metrics=True) as run:  # Start a new run
            #loaded_model = load_model(deployment=None)
            run_id = run.info.run_id  # Get the run ID
            print(f"Run ID: {run_id}")
            log_data_files()
            print("Evaluating the model...")
            X_test_st = pd.read_csv(X_test_file).values
            y_test = pd.read_csv(y_test_file).values.ravel()

            # Evaluate the model and log metrics
            accuracy, precision, recall, f1, cm = evaluate_model(
                svm_model, X_test_st, y_test
            )

# Improvement logic
if args.improve:
    svm_model=load_model(deployment=None)
    if svm_model is None:
        print("No trained model available. Run training first.")
    else:
        mlflow.end_run()
        with mlflow.start_run(
            run_name="SVM_Improvement", log_system_metrics=True
        ) as run:
            print("Improving the model...")
            X_train_st = pd.read_csv(X_train_file).values
            X_test_st = pd.read_csv(X_test_file).values
            y_train = pd.read_csv(y_train_file).values.ravel()
            y_test = pd.read_csv(y_test_file).values.ravel()

            # Improve the model
            best_model, best_params = improve_model(
                X_train_st, y_train, X_test_st, y_test
            )
            svm_model = best_model
            run_id = run.info.run_id

            # Log best hyperparameters
            mlflow.log_params(best_params)
            log_system_metrics_function()  # Log system metrics
            

            # Evaluate the improved model and log artifacts
            evaluate_model(svm_model, X_test_st, y_test, model_name="model")

            # Save the improved model
            save_model(svm_model)
            # Register the model in the Model Registry
            model_uri = f"runs:/{run_id}/model"
            model_name = "Churn_Prediction_Model"
            mlflow.register_model(model_uri, model_name)
            print(f"Model registered as '{model_name}'.")

# Save logic
if args.save:
    if svm_model is None:
        print("No trained model available to save.")
    else:
        mlflow.end_run()
        with mlflow.start_run(run_name="SVM_Saving"):
            print("Saving the model...")
            save_model(svm_model)
            mlflow.sklearn.log_model(svm_model, "model")
            log_data_files()
            log_system_metrics_function()


# Load and evaluate saved model
if args.load:
    if os.path.exists(model_path):
        mlflow.end_run()
        with mlflow.start_run(
            run_name="SVM_Loading_Evaluation"):
            print("Loading the saved model...")
            loaded_model = load_model(deployment=None)
            X_test_st = pd.read_csv(X_test_file).values
            y_test = pd.read_csv(y_test_file).values.ravel()
            

            evaluate_model(loaded_model, X_test_st, y_test)
    else:
        print("No saved model found. Train and save a model first.")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
     

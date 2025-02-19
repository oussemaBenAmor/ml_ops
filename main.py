import argparse
import os
import pandas as pd
import joblib
import numpy as np
import mlflow
import mlflow.sklearn 
import matplotlib.pyplot as plt
import seaborn as sns
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    improve_model,
    save_model,
    load_model,
)

mlflow.set_tracking_uri("http://localhost:5000")
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


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train, evaluate, and improve an SVM model.")
parser.add_argument("--prepare", action="store_true", help="Prepare and save data")
parser.add_argument("--train", action="store_true", help="Train the model")
parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
parser.add_argument("--improve", action="store_true", help="Improve the model")
parser.add_argument("--save", action="store_true", help="Save the model")
parser.add_argument("--load", action="store_true", help="Load the saved model and evaluate")

args = parser.parse_args()

# Data preparation logic
if args.prepare:
	mlflow.end_run() 
	with mlflow.start_run(run_name="SVM_preaparing_data"):
		scaler, label_encoders = prepare_data(train_path, test_path)
		print("Data preparation completed and saved.")

# Check if prepared data exists before proceeding with other steps
	if os.path.exists(X_train_file) and os.path.exists(X_test_file):
    # Load saved data
		X_train_st = pd.read_csv(X_train_file).values
		X_test_st = pd.read_csv(X_test_file).values
		y_train = pd.read_csv(y_train_file).values.ravel()
		y_test = pd.read_csv(y_test_file).values.ravel()
    
  
	else:
		print("Prepared data not found. Run --prepare first.")
		exit()

svm_model = None

# Load existing model if available
if os.path.exists(model_path):
	print("Existing model found. Loading...")
	svm_model = load_model()
else:
	print("No existing model found.")

# Training logic
if args.train:
	mlflow.end_run() 
	with mlflow.start_run(run_name="SVM_training"):
		print("Training the model...")
		X_train_st = pd.read_csv(X_train_file).values

		y_train = pd.read_csv(y_train_file).values.ravel()

		svm_model = train_model(X_train_st, y_train)
		save_model(svm_model)
		print("Model training complete.")




# Evaluation logic (only use existing model or trained one)
if args.evaluate:
    if svm_model is None:
        print("No trained model available. Run training first.")
    else:
        mlflow.end_run()  
        with mlflow.start_run(run_name="SVM_Evaluation"):
            loaded_model = load_model()
            print("Evaluating the model...")
            X_test_st = pd.read_csv(X_test_file).values
            y_test = pd.read_csv(y_test_file).values.ravel()
            
            accuracy, precision, recall,f1,cm = evaluate_model(loaded_model, X_test_st, y_test)
            
            # Log the evaluation metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1 score", f1)
            
            # Optionally log confusion matrix as an artifact

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.title("Confusion Matrix")

            # Save the confusion matrix plot
            cm_image_path = "confusion_matrix.png"
            plt.savefig(cm_image_path)

            # Log confusion matrix artifact
            mlflow.log_artifact(cm_image_path)
            print(f"Confusion matrix image saved and logged as artifact: {cm_image_path}")

# Improvement logic (uses an existing trained model)
# Improvement logic (uses an existing trained model)
if args.improve:
    if svm_model is None:
        print("No trained model available. Run training first.")
    else:
        # Run the grid search and improve the model
        print("Improving the model...")

        # Load the previously saved data
        X_train_st = pd.read_csv(X_train_file).values
        X_test_st = pd.read_csv(X_test_file).values
        y_train = pd.read_csv(y_train_file).values.ravel()
        y_test = pd.read_csv(y_test_file).values.ravel()

        # Improve the model and get the best model and its parameters
        best_model, best_params = improve_model(X_train_st, y_train, X_test_st, y_test)
        svm_model = best_model

        # Now that the model improvement is done, start the MLflow run
        with mlflow.start_run(run_name="SVM_Model_Improvement"):
            # Log parameters of the improved model
            mlflow.log_param("C", best_params['C'])
            mlflow.log_param("kernel", best_params['kernel'])
            mlflow.log_param("degree", best_params['degree'])

            # Log metrics of the improved model
            accuracy, precision, recall, f1, confusion_matrix = evaluate_model(svm_model, X_test_st, y_test)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1 score", f1)

            # Save the improved model and log it with MLflow
            save_model(svm_model)
            mlflow.sklearn.log_model(svm_model, "best_model")

            # Log confusion matrix as an artifact
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.title("Confusion Matrix")

            # Save the confusion matrix plot
            cm_image_path = "confusion_matrix.png"
            plt.savefig(cm_image_path)

            # Log confusion matrix artifact
            mlflow.log_artifact(cm_image_path)
            print(f"Confusion matrix image saved and logged as artifact: {cm_image_path}")

       
# Save logic (only saves if a model is available)
if args.save:
    if svm_model is None:
        print("No trained model available to save.")
    else:
        print("Saving the model...")
        save_model(svm_model)
        
        with mlflow.start_run(run_name="SVM_Model_Saving"):
            mlflow.sklearn.log_model(svm_model, "final_saved_model")
            # You can log metrics if you want
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1_score)
# Load and evaluate saved model
if args.load:
	if os.path.exists(model_path):
		print("Loading the saved model...")
		loaded_model = load_model()
		with mlflow.start_run(run_name="SVM_Loading_Evaluation"):
			X_test_st = pd.read_csv(X_test_file).values
			y_test = pd.read_csv(y_test_file).values.ravel()
			accuracy, precision, recall = evaluate_model(loaded_model, X_test_st,y_test)
			mlflow.log_metric("accuracy", accuracy)
			mlflow.log_metric("precision", precision)
			mlflow.log_metric("recall", recall)
	else:
		print("No saved model found. Train and save a model first.")


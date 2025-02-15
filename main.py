import argparse
import os
import pandas as pd
import joblib
import numpy as np
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    improve_model,
    save_model,
    load_model,
)

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
		loaded_model = load_model()
		print("Evaluating the model...")

		X_test_st = pd.read_csv(X_test_file).values

		y_test = pd.read_csv(y_test_file).values.ravel()
		evaluate_model(loaded_model, X_test_st, y_test)

# Improvement logic (uses an existing trained model)
if args.improve:
	if svm_model is None:
		print("No trained model available. Run training first.")
	else:
		print("Improving the model...")
		X_train_st = pd.read_csv(X_train_file).values
		X_test_st = pd.read_csv(X_test_file).values
		y_train = pd.read_csv(y_train_file).values.ravel()
		y_test = pd.read_csv(y_test_file).values.ravel()
		best_model = improve_model(X_train_st, y_train, X_test_st, y_test)
		svm_model = best_model
		save_model(svm_model)

# Save logic (only saves if a model is available)
if args.save:
	if svm_model is None:
		print("No trained model available to save.")
	else:
		print("Saving the model...")
		save_model(svm_model)

# Load and evaluate saved model
if args.load:
	if os.path.exists(model_path):
		print("Loading the saved model...")
		loaded_model = load_model()
		evaluate_model(loaded_model, X_test_st, y_test)
	else:
		print("No saved model found. Train and save a model first.")


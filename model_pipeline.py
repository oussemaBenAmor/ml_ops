import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import statsmodels.api as sm
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV

import joblib


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
    return  scaler, label_encoders


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
    return svm_model


def evaluate_model(model, X_test, y_test):
    # Make predictions
	y_pred = model.predict(X_test)

    # Generate confusion matrix
	cm = confusion_matrix(y_test, y_pred)
	print("Confusion Matrix:")
	print(cm)

    # Compute and print evaluation metrics
	accuracy = accuracy_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)

	print("Accuracy:", accuracy)
	print("Precision:", precision)
	print("Recall:", recall)
	print("F1 Score:", f1)

    # Return the metrics as a tuple
	return accuracy, precision, recall,f1,cm


def improve_model(X_train, y_train, X_test, y_test):

    print("# Tuning hyper-parameters for F1 Score")

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

    # Recherche des meilleurs paramètres
    clf = GridSearchCV(SVC(), tuned_parameters, scoring="f1", cv=5)
    clf.fit(X_train, y_train)

    print("Best Parameters:", clf.best_params_)
    print("Best Cross-Validation Score:", clf.best_score_)

    # Meilleur modèle après GridSearchCV
    best_svm = clf.best_estimator_
    best_params = clf.best_params_

    # Prédiction avec le meilleur modèle
    y_pred_best_svm = best_svm.predict(X_test)

    # Affichage de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred_best_svm)
    print("Confusion Matrix:")
    print(cm)

    # Affichage des métriques
    print("Accuracy:", accuracy_score(y_test, y_pred_best_svm))
    print("Precision:", precision_score(y_test, y_pred_best_svm, zero_division=1))
    print("Recall:", recall_score(y_test, y_pred_best_svm, zero_division=1))
    print("F1 Score:", f1_score(y_test, y_pred_best_svm, zero_division=1))

    return best_svm , best_params


def save_model(model, filename="best_svm_model.pkl"):

    joblib.dump(model, filename)
    print(f" Model saved in file :: '{filename}'.")


def load_model(filename="best_svm_model.pkl"):

    model = joblib.load(filename)
    print(f" Model loaded from '{filename}'.")
    return model

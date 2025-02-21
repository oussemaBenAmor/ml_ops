from flask import request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump
from sklearn.preprocessing import StandardScaler


# Load initial model and scaler
model = joblib.load('best_svm_model.pkl')
scaler = joblib.load('scaler.pkl')

def configure_routes(app):
    @app.route('/')
    def default():
        return render_template('home.html')
    
    @app.route('/home')
    def home():
        return render_template('home.html')

    @app.route('/team')
    def team():
        return render_template('team.html')

    @app.route('/svm')
    def SVM_open():
        return render_template('svm.html')

    @app.route("/newmodel")
    def retrain_page():
        return render_template("newModel.html")

    @app.route("/retrain", methods=["POST"])
    def retrain():
        try:
            # Load dataset
            x_train = pd.read_csv("X_train.csv")
            x_test = pd.read_csv("X_test.csv")
            y_train = pd.read_csv("y_train.csv")
            y_test = pd.read_csv("y_test.csv")

            # Get hyperparameters from the request
            C_values = float(request.form.get("C", 1.0))  # Default C = 1.0
            kernel_type = request.form.get("kernel", "rbf")  # Default kernel = "rbf"
            degree_value = int(request.form.get("degree", 3))  # Default degree = 3 (for poly kernel)
            gamma_value = request.form.get("gamma", "scale")  # Default gamma = "scale"
            coef0_value = float(request.form.get("coef0", 0.0))  # Default coef0 = 0.0
            random_state = int(request.form.get("random_state", 42))  # Default random state

            # Initialize and train the SVM model
            model = SVC(
                C=C_values,
                kernel=kernel_type,
                degree=degree_value,
                gamma=gamma_value,
                coef0=coef0_value,
                random_state=random_state,
            )

            model.fit(x_train, y_train.values.ravel())  # Ensure y_train is a 1D array

            # Evaluate the model
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            # Save the model
            dump(model, "new_model.joblib")

            # Return response
            return jsonify({
                "message": "SVM model retrained successfully!",
                "C": C_values,
                "kernel": kernel_type,
                "degree": degree_value,
                "gamma": gamma_value,
                "coef0": coef0_value,
                "random_state": random_state,
                "accuracy": f"{accuracy:.2f}",
                "precision": f"{precision:.2f}",
                "recall": f"{recall:.2f}",
                "f1": f"{f1:.2f}",
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/predict', methods=['POST'])
    def predict():
        if request.method == 'POST':
            selected_model_name = request.form.get('model_select', '')

            # Load the appropriate model based on selection
            model = None
            try:
                if selected_model_name == "svm":
                    model = joblib.load('best_svm_model.pkl')
                elif selected_model_name == "new_model":
                    model = joblib.load('new_model.joblib')
                else:
                    return jsonify({"error": "Invalid model selected."}), 400

                # Getting the form data
                international_plan = int(request.form['international_plan'])
                number_vmail_messages = float(request.form['number_vmail_messages'])
                total_day_minutes = float(request.form['total_day_minutes'])
                total_eve_minutes = float(request.form['total_eve_minutes'])
                total_night_minutes = float(request.form['total_night_minutes'])
                total_intl_minutes = float(request.form['total_intl_minutes'])
                total_intl_calls = int(request.form['total_intl_calls'])
                customer_service_calls = int(request.form['customer_service_calls'])

                # Prepare the input data for prediction (ensure features align)
                data = {
                    'International plan': [international_plan],
                    'Number vmail messages': [number_vmail_messages],
                    'Total day minutes': [total_day_minutes],
                    'Total eve minutes': [total_eve_minutes],
                    'Total night minutes': [total_night_minutes],
                    'Total intl minutes': [total_intl_minutes],
                    'Total intl calls': [total_intl_calls],
                    'Customer service calls': [customer_service_calls]
                }

                input_df = pd.DataFrame(data)
                input_df = scaler.transform(input_df)  # Ensure that `scaler` was correctly defined

                # Make predictions
                prediction = model.predict(input_df)

                if prediction.size == 0:
                    return jsonify({"error": "No prediction made."}), 500
                
                prediction = prediction[0]

                # Result logic
                result = "The customer will not churn." if prediction == 0 else "The customer is likely to churn."

                return jsonify({'result': result})
            
            except Exception as e:
                return jsonify({"error": str(e)}), 500
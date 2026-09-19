import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained pipeline
model = joblib.load("final_pipeline.pkl")


# Features required by the model
required_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


# Numeric features
numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


# Categorical features
categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# Home / Health-check route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Check whether request contains JSON data
    if not data:
        return jsonify({
            "error": "Request body must contain valid JSON data"
        }), 400


    # Check whether all required features are present
    missing_features = [
        feature for feature in required_features
        if feature not in data
    ]

    if missing_features:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing_features
        }), 400


    # Convert incoming JSON into a DataFrame
    # Keep only the features required by the model
    input_data = pd.DataFrame([data])[required_features]


    # Validate numeric features
    for feature in numeric_features:

        try:
            input_data[feature] = pd.to_numeric(
                input_data[feature]
            )

        except (ValueError, TypeError):

            return jsonify({
                "error": f"Invalid numeric value for {feature}"
            }), 400


    # Validate categorical features
    for feature in categorical_features:

        # Get the trained OneHotEncoder
        encoder = (
            model
            .named_steps["preprocessor"]
            .named_transformers_["cat"]
        )

        # Find the position of this feature
        feature_index = list(categorical_features).index(feature)

        # Get values learned during training
        allowed_values = encoder.categories_[feature_index]

        # Check customer's value
        if input_data[feature].iloc[0] not in allowed_values:

            return jsonify({
                "error": f"Invalid value for {feature}",
                "allowed_values": list(allowed_values)
            }), 400


    # Handle unexpected prediction errors
    try:

        # Get churn probability
        probability = model.predict_proba(
            input_data
        )[:, 1][0]

        # Apply selected threshold
        prediction = (
            "Churn"
            if probability >= 0.33
            else "No Churn"
        )

    except Exception:

        return jsonify({
            "error": "An error occurred while making the prediction"
        }), 500


    # Return successful prediction
    return jsonify({
        "churn_probability": probability,
        "prediction": prediction
    })


# Start Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
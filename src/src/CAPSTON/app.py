from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("svm_dos_detector.pkl")

# Get model feature names
expected_features = model.feature_names_in_

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "DoS Detection API is Running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        df = pd.DataFrame(data)

        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0  # Fill missing features with 0

        # Predict
        prediction = model.predict(df[expected_features])

        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

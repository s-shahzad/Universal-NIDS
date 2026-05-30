import joblib

# Load the trained model
model = joblib.load("svm_dos_detector.pkl")

# Print expected features
print("Model was trained with these features:", model.feature_names_in_)

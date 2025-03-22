from joblib import load

# Load the model
loaded_model = load('water_quality_model.joblib')
print("Model loaded using joblib.")

# Example usage
prediction = loaded_model.predict([[7.5, 3.2, 8.1, 5.4]])
print("Prediction:", prediction)

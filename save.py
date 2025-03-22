from joblib import dump

# Save the model
dump(model, 'water_quality_model.joblib')
print("Model saved using joblib.")

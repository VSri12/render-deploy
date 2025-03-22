import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data_path = r"C:\Users\user\Downloads\water_potability.csv"
data = pd.read_csv(data_path)

# Drop rows with missing values (optional, depending on the dataset quality)
data = data.dropna()

# Features and target
X = data[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
          'Organic_carbon', 'Trihalomethanes', 'Turbidity']]  # Adjust based on available features
y = data['Potability']  # Assuming 'Potability' is the target column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model (optional)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model as a .pkl file
model_file = 'water_potability_model.pkl'
joblib.dump(model, model_file)
print(f"Model saved as {model_file}")

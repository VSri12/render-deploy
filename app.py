from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('water_potability_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')
@app.route("/about")
def about():
    return render_template("about.html")
@app.route('/main', methods=['GET', 'POST'])
def main_page():
    result = None
    if request.method == 'POST':
        try:
            data = {
                'ph': float(request.form['ph']),
                'Hardness': float(request.form['Hardness']),
                'Solids': float(request.form['Solids']),
                'Chloramines': float(request.form['Chloramines']),
                'Sulfate': float(request.form['Sulfate']),
                'Conductivity': float(request.form['Conductivity']),
                'Organic_carbon': float(request.form['Organic_carbon']),
                'Trihalomethanes': float(request.form['Trihalomethanes']),
                'Turbidity': float(request.form['Turbidity'])
            }

            features = np.array([[data['ph'], data['Hardness'], data['Solids'], data['Chloramines'],
                                  data['Sulfate'], data['Conductivity'], data['Organic_carbon'],
                                  data['Trihalomethanes'], data['Turbidity']]])

            prediction = model.predict(features)
            result = "Potable" if prediction[0] == 1 else "Not Potable"
        except Exception as e:
            result = f"Error: {e}"

    return render_template('main.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

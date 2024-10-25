from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model from the .pkl file
app = Flask(__name__)
model = pickle.load(open('gb_model.pkl', 'rb'))

# Define mappings for label-encoded values
vehicle_mapping = {0: "Car", 1: "Truck", 2: "Motorcycle", 3: "Bicycle", 4: "Pedestrian"}
weather_mapping = {0: "Clear", 1: "Rain", 2: "Fog", 3: "Snow"}
cause_mapping = {0: "Over Speeding", 1: "Drunk Driving", 2: "Distracted Driving", 3: "Mechanical Failure", 4: "Road Conditions"}
severity_mapping = {0: "Low", 1: "Moderate", 2: "Severe"}  # For Injury Severity predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        vehicle_involved = int(request.form['vehicle_involved'])
        weather_conditions = int(request.form['weather_conditions'])
        cause = int(request.form['cause'])
        num_vehicles = int(request.form['num_vehicles'])
        num_injuries = int(request.form['num_injuries'])
        num_fatalities = int(request.form['num_fatalities'])

        # Create feature array for prediction
        features = np.array([[vehicle_involved, weather_conditions, cause, num_vehicles, num_injuries, num_fatalities]])

        # Make prediction (integer output for injury severity)
        prediction = model.predict(features)
        severity_level = severity_mapping[int(prediction[0])]  # Convert to descriptive string

        # Convert input values to descriptive strings
        vehicle_type = vehicle_mapping[vehicle_involved]
        weather_desc = weather_mapping[weather_conditions]
        cause_desc = cause_mapping[cause]

        # Pass the results to result.html
        return render_template(
            'result.html',
            prediction=severity_level,
            vehicle=vehicle_type,
            weather=weather_desc,
            cause=cause_desc,
            num_vehicles=num_vehicles,
            num_injuries=num_injuries,
            num_fatalities=num_fatalities
        )

    except Exception as e:
        # Redirect to error page in case of an exception
        return render_template('error.html')

if __name__ == "__main__":
    app.run(debug=True)

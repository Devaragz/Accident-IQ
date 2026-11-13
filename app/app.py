"""
Flask Web Application for   Accident-IQ
"""

import warnings

warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import AccidentSeverityPredictor

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')


# Load the trained model
def load_model():
    """Load the trained model"""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.pkl')
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Global model variable
model = load_model()

# Initialize predictor
predictor = None
if model:
    try:
        predictor = AccidentSeverityPredictor(
            os.path.join(os.path.dirname(__file__), '..', 'best_model.pkl')
        )
    except Exception as e:
        print(f"Error initializing predictor: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction="Error: Model not found")

    try:
        # Get form data
        weather = request.form.get('weather', 'Clear')
        light = request.form.get('light', 'Daylight')
        road_surface = request.form.get('road_surface', 'Dry')
        road_type = request.form.get('road_type', 'Highway')
        vehicles = int(request.form.get('vehicles', 2))
        killed = int(request.form.get('killed', 0))
        injured = int(request.form.get('injured', 1))
        year = int(request.form.get('year', 2022))

        # Create input with exactly 31 features (excluding target)
        # We'll use default/typical values for features not in the form
        # Feature order must match training data exactly
        input_data = {
            'Time': 0,  # Default encoded time
            'Day_of_week': 0,  # Default: Monday
            'Age_band_of_driver': 2,  # Default: middle age
            'Sex_of_driver': 0,  # Default: Male
            'Educational_level': 3,  # Default: Secondary
            'Vehicle_driver_relation': 0,  # Default: Owner
            'Driving_experience': 3,  # Default: 5-10 years
            'Type_of_vehicle': 0,  # Default: Car
            'Owner_of_vehicle': 0,  # Default: Owner
            'Service_year_of_vehicle': 2,  # Default: 2-5 years
            'Defect_of_vehicle': 2,  # Default: No defect
            'Area_accident_occured': 6,  # Default: Urban
            'Lanes_or_Medians': 4,  # Default: Two-way
            'Road_allignment': 5,  # Default: Straight
            'Types_of_Junction': 1,  # Default: No junction
            'Road_surface_type': 0,  # Will be filled from form
            'Road_surface_conditions': 0,  # Will be filled from form
            'Light_conditions': 3,  # Will be filled from form
            'Weather_conditions': 2,  # Will be filled from form
            'Type_of_collision': 0,  # Default: Vehicle-vehicle
            'Number_of_vehicles_involved': vehicles,
            'Number_of_casualties': injured + killed,
            'Vehicle_movement': 2,  # Default: Going straight
            'Casualty_class': 3,  # Default: Driver
            'Sex_of_casualty': 2,  # Default: Male
            'Age_band_of_casualty': 5,  # Default: Middle age
            'Casualty_severity': 3,  # Default: Minor
            'Work_of_casuality': 0,  # Default: Employee
            'Fitness_of_casuality': 2,  # Default: Normal
            'Pedestrian_movement': 5,  # Default: Not pedestrian
            'Cause_of_accident': 10  # Default: Driver error
        }

        # Map form inputs to encoded values based on actual dataset
        weather_map = {'Clear': 2, 'Rainy': 4, 'Foggy': 1, 'Snowy': 6, 'Cloudy': 0}
        light_map = {'Daylight': 0, 'Night': 3, 'Dawn/Dusk': 1, 'Street Lights': 2}
        surface_map = {'Dry': 0, 'Wet': 3, 'Slippery': 2, 'Icy': 1, 'Muddy': 1}

        # Update with form values
        input_data['Weather_conditions'] = weather_map.get(weather, 2)
        input_data['Light_conditions'] = light_map.get(light, 3)
        input_data['Road_surface_conditions'] = surface_map.get(road_surface, 0)
        input_data['Road_surface_type'] = surface_map.get(road_surface, 0)

        # Make prediction using the predictor
        if predictor:
            predicted_severity = predictor.predict_severity(input_data)
        else:
            # Fallback to direct model prediction
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            severity_mapping = {0: 'Fatal', 1: 'Serious', 2: 'Minor'}
            predicted_severity = severity_mapping.get(prediction, 'Unknown')

        return render_template('index.html', prediction=predicted_severity)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8501)
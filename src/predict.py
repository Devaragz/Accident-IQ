"""
Prediction Module - functionality to make predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_model

class AccidentSeverityPredictor:
    """
    Predictor class for making accident severity predictions
    """
    
    def __init__(self, model_path, label_encoders_path=None):
        """
        Initialize the predictor
        """
        self.model = load_model(model_path)
        self.label_encoders = None
        self.severity_mapping = {
            0: 'Fatal',
            1: 'Serious',
            2: 'Minor'
        }
        
        if label_encoders_path and os.path.exists(label_encoders_path):
            self.label_encoders = joblib.load(label_encoders_path)
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction
        """
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Apply label encoding if encoders are available
        if self.label_encoders:
            for col, encoder in self.label_encoders.items():
                if col in input_df.columns and col != 'Accident_Severity':
                    try:
                        input_df[col] = encoder.transform(input_df[col].astype(str))
                    except:
                        # Handle unknown categories
                        input_df[col] = 0
        
        return input_df
    
    def predict_severity(self, input_data, return_proba=False):
        """
        Predict accident severity
        """
        # Preprocess input
        processed_input = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(processed_input)[0]
        severity = self.severity_mapping.get(prediction, 'Unknown')
        
        if return_proba:
            probabilities = self.model.predict_proba(processed_input)[0]
            prob_dict = {
                self.severity_mapping[i]: prob 
                for i, prob in enumerate(probabilities)
            }
            return severity, prob_dict
        
        return severity
    
    def predict_batch(self, input_data_list):
        """
        Make predictions for multiple inputs
        """
        if isinstance(input_data_list, pd.DataFrame):
            processed_input = self.preprocess_input(input_data_list)
        else:
            processed_input = self.preprocess_input(pd.DataFrame(input_data_list))
        
        predictions = self.model.predict(processed_input)
        severities = [self.severity_mapping.get(pred, 'Unknown') for pred in predictions]
        
        return severities
    
    def explain_prediction(self, input_data):
        """
        Provide explanation for the prediction
        """
        severity, probabilities = self.predict_severity(input_data, return_proba=True)
        
        explanation = {
            'predicted_severity': severity,
            'confidence': max(probabilities.values()),
            'probabilities': probabilities,
            'input_features': input_data
        }
        return explanation


def predict_from_console():
    """
    Interactive console-based prediction
    """
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'best_model.pkl')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("Error: Model not found! Please train the model first.")
        return
    
    # Initialize predictor
    predictor = AccidentSeverityPredictor(model_path)
    
    print("\n" + "="*60)
    print("ROAD ACCIDENT SEVERITY PREDICTOR")
    print("="*60)
    
    print("\nEnter accident details:")
    
    # Example input collection
    input_data = {}
    
    print("\nAvailable options for categorical features:")
    print("Weather: Clear, Rainy, Foggy, Snowy")
    print("Road Type: Highway, Urban Road, Rural Road")
    print("Light Condition: Daylight, Night, Dawn/Dusk")
    print("Road Surface: Dry, Wet, Slippery, Icy")
    
    try:
        # Collect input (customize based on your features)
        input_data['Weather_Condition'] = input("\nWeather Condition: ")
        input_data['Road_Type'] = input("Road Type: ")
        input_data['Light_Condition'] = input("Light Condition: ")
        input_data['Road_Surface'] = input("Road Surface: ")
        input_data['Vehicles_Involved'] = int(input("Number of Vehicles Involved: "))
        input_data['Persons_Killed'] = int(input("Persons Killed: "))
        input_data['Persons_Injured'] = int(input("Persons Injured: "))
        
        # Make prediction
        print("\n" + "="*60)
        print("Making prediction...")
        
        explanation = predictor.explain_prediction(input_data)
        
        print("\n=== PREDICTION RESULTS ===")
        print(f"Predicted Severity: {explanation['predicted_severity']}")
        print(f"Confidence: {explanation['confidence']*100:.2f}%")
        print("\nProbabilities:")
        for severity, prob in explanation['probabilities'].items():
            print(f"  {severity}: {prob*100:.2f}%")
        
    except Exception as e:
        print(f"\nError making prediction: {e}")
        print("Please ensure all inputs are valid.")


def predict_from_file(input_file, output_file=None):
    """
    Make predictions from a CSV file
    """
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'best_model.pkl')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("Error: Model not found! Please train the model first.")
        return
    
    # Initialize predictor
    predictor = AccidentSeverityPredictor(model_path)
    
    # Load input data
    print(f"Loading data from {input_file}...")
    input_df = pd.read_csv(input_file)
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict_batch(input_df)
    
    # Add predictions to dataframe
    output_df = input_df.copy()
    output_df['Predicted_Severity'] = predictions
    
    # Save results
    if output_file:
        output_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    else:
        print("\n=== Predictions ===")
        print(output_df[['Predicted_Severity']].head(10))
    
    return output_df


def main():
    """
    Main function for prediction module
    """
    print("\n" + "="*60)
    print("ACCIDENT SEVERITY PREDICTION MODULE")
    print("="*60)
    
    print("\nOptions:")
    print("1. Interactive prediction (console input)")
    print("2. Batch prediction from file")
    
    choice = input("\nSelect option (1 or 2): ")
    
    if choice == '1':
        predict_from_console()
    elif choice == '2':
        input_file = input("Enter path to input CSV file: ")
        output_file = input("Enter path to save predictions (or press Enter to skip): ")
        
        if not output_file:
            output_file = None
        
        predict_from_file(input_file, output_file)
    else:
        print("Invalid option!")


if __name__ == "__main__":
    main()

"""
Model Training Module - Trains multiple ML models
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import joblib
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import DataPreprocessor
from src.utils import save_model, evaluate_model


class ModelTrainer:
    """
    Model Training Class for training and tuning multiple ML models
    """
    
    def __init__(self):
        """Initialize the model trainer"""
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.results = []
        
    def initialize_models(self):
        """Initialize all models with default parameters"""
        print("\n=== Initializing Models ===")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=10
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric='mlogloss'
            )
        }
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
        
        return self.models
    
    def train_model(self, model_name, X_train, y_train):
        """
        Train a single model
        """
        print(f"\nTraining {model_name}...")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        print(f"{model_name} training completed!")
        
        return model
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train all models and evaluate them
        """
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        trained_models = {}
        
        for model_name in self.models.keys():
            # Train model
            trained_model = self.train_model(model_name, X_train, y_train)
            trained_models[model_name] = trained_model
            
            # Evaluate model
            results = evaluate_model(trained_model, X_test, y_test, model_name)
            self.results.append(results)
            
            # Track best model
            if results['Accuracy'] > self.best_score:
                self.best_score = results['Accuracy']
                self.best_model = trained_model
                self.best_model_name = model_name
        
        print("\n" + "="*60)
        print(f"BEST MODEL: {self.best_model_name} (Accuracy: {self.best_score:.4f})")
        print("="*60)
        
        return trained_models
    
    def perform_cross_validation(self, model_name, X, y, cv=5):
        """
        Perform cross-validation on a model
        """
        print(f"\n=== Cross-Validation for {model_name} ===")
        
        model = self.models[model_name]
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        print(f"CV Scores: {scores}")
        print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def get_results_dataframe(self):
        """
        Get results as a DataFrame
        """
        results_df = pd.DataFrame(self.results)
        return results_df
    
    def save_best_model(self, output_path):
        """
        Save the best performing model
        """
        if self.best_model is not None:
            save_model(self.best_model, output_path)
            print(f"\nBest model ({self.best_model_name}) saved successfully!")
        else:
            print("No model has been trained yet!")


def main():
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'dataset', 'processed_data.csv')
    model_output_path = os.path.join(project_root, 'best_model.pkl')
    results_output_path = os.path.join(project_root, 'results', 'model_performance.csv')
    
    # Check if processed data exists
    if not os.path.exists(data_path):
        print("Processed data not found. Running preprocessing first...")
        raw_data_path = os.path.join(project_root, 'dataset', 'road_accidents_india_2017_2022.csv')
        preprocessor = DataPreprocessor(raw_data_path)
        preprocessor.preprocess_pipeline(save_processed=True, output_path=data_path)
    
    # Load preprocessed data
    print("\n=== Loading Preprocessed Data ===")
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape}")
    
    # Prepare data for modeling
    preprocessor = DataPreprocessor(data_path)
    preprocessor.df = df
    preprocessor.target_column = 'Accident_severity'  # Fixed: lowercase 's'
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_data_for_modeling(
        test_size=0.2,
        random_state=42
    )
    
    # Initialize trainer
    trainer = ModelTrainer()
    trainer.initialize_models()
    
    # Train all models
    trained_models = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Get results
    results_df = trainer.get_results_dataframe()
    print("\n=== Model Performance Summary ===")
    print(results_df.to_string(index=False))
    
    # Save results
    os.makedirs(os.path.dirname(results_output_path), exist_ok=True)
    results_df.to_csv(results_output_path, index=False)
    print(f"\nResults saved to: {results_output_path}")
    
    # Save best model
    trainer.save_best_model(model_output_path)
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return trainer


if __name__ == "__main__":
    trainer = main()

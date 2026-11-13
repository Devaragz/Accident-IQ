"""
Model Evaluation Module - model evaluation and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve
)
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    load_model, plot_confusion_matrix, plot_feature_importance,
    compare_models, evaluate_model
)


class ModelEvaluator:
    """
    Model Evaluation Class for comprehensive model assessment
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the evaluator
        """
        self.model = None
        self.model_path = model_path
        self.predictions = None
        self.evaluation_results = {}
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
    
    def load_evaluation_model(self, model_path):
        """
        Load a model for evaluation
        """
        self.model = load_model(model_path)
        self.model_path = model_path
        return self.model
    
    def predict(self, X_test):
        """
        Make predictions using the loaded model
        """
        if self.model is None:
            raise ValueError("No model loaded! Please load a model first.")
        
        self.predictions = self.model.predict(X_test)
        return self.predictions
    
    def evaluate_classification(self, y_true, y_pred=None):
        """
        Evaluate classification performance
        """
        if y_pred is None:
            y_pred = self.predictions
        
        if y_pred is None:
            raise ValueError("No predictions available! Please run predict() first.")
        
        print("\n=== Classification Evaluation ===")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Store results
        self.evaluation_results = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        # Print results
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        return self.evaluation_results
    
    def print_classification_report(self, y_true, y_pred=None, target_names=None):
        """
        Print detailed classification report
        """
        if y_pred is None:
            y_pred = self.predictions
        
        print("\n=== Detailed Classification Report ===")
        print(classification_report(y_true, y_pred, target_names=target_names))
    
    def plot_confusion_matrix_custom(self, y_true, y_pred=None, labels=None, save_path=None):
        """
        Plot confusion matrix
        """
        if y_pred is None:
            y_pred = self.predictions
        
        plot_confusion_matrix(y_true, y_pred, labels=labels, save_path=save_path)
    
    def plot_class_distribution(self, y_true, y_pred=None, labels=None, save_path=None):
        """
        Plot distribution of actual vs predicted classes
        """
        if y_pred is None:
            y_pred = self.predictions
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Actual distribution
        unique, counts = np.unique(y_true, return_counts=True)
        axes[0].bar(unique, counts, color='skyblue', edgecolor='black')
        axes[0].set_title('Actual Class Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Class', fontsize=10)
        axes[0].set_ylabel('Count', fontsize=10)
        if labels:
            axes[0].set_xticks(unique)
            axes[0].set_xticklabels(labels)
        
        # Predicted distribution
        unique, counts = np.unique(y_pred, return_counts=True)
        axes[1].bar(unique, counts, color='lightcoral', edgecolor='black')
        axes[1].set_title('Predicted Class Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Class', fontsize=10)
        axes[1].set_ylabel('Count', fontsize=10)
        if labels:
            axes[1].set_xticks(unique)
            axes[1].set_xticklabels(labels)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_importance_custom(self, feature_names, top_n=15, save_path=None):
        """
        Plot feature importance for the loaded model
        """
        if self.model is None:
            raise ValueError("No model loaded!")
        
        plot_feature_importance(self.model, feature_names, top_n=top_n, save_path=save_path)
    
    def analyze_misclassifications(self, X_test, y_true, y_pred=None, feature_names=None):
        """
        Analyze misclassified samples
        """
        if y_pred is None:
            y_pred = self.predictions
        
        # Find misclassified samples
        misclassified_mask = y_true != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]
        
        print(f"\n=== Misclassification Analysis ===")
        print(f"Total misclassified samples: {len(misclassified_indices)}")
        print(f"Misclassification rate: {len(misclassified_indices) / len(y_true) * 100:.2f}%")
        
        if len(misclassified_indices) > 0:
            # Create DataFrame of misclassified samples
            if hasattr(X_test, 'values'):
                X_test_array = X_test.values
            else:
                X_test_array = X_test
            
            misclassified_df = pd.DataFrame(
                X_test_array[misclassified_indices],
                columns=feature_names if feature_names else range(X_test_array.shape[1])
            )
            misclassified_df['True_Label'] = y_true.iloc[misclassified_indices].values if hasattr(y_true, 'iloc') else y_true[misclassified_indices]
            misclassified_df['Predicted_Label'] = y_pred[misclassified_indices]
            
            print(f"\nFirst 5 misclassified samples:")
            print(misclassified_df.head())
            
            return misclassified_df
        
        return None
    
    def calculate_per_class_metrics(self, y_true, y_pred=None, target_names=None):
        """
        Calculate metrics for each class
        """
        if y_pred is None:
            y_pred = self.predictions
        
        # Calculate per-class metrics
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        
        # Create DataFrame
        metrics_df = pd.DataFrame({
            'Class': target_names if target_names else list(range(len(precision))),
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        print("\n=== Per-Class Metrics ===")
        print(metrics_df.to_string(index=False))
        
        return metrics_df
    
    def save_evaluation_report(self, y_true, y_pred=None, output_path=None):
        """
        Save comprehensive evaluation report
        """
        if y_pred is None:
            y_pred = self.predictions
        
        if output_path is None:
            output_path = 'evaluation_report.txt'
        
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Overall metrics
            f.write("Overall Metrics:\n")
            f.write(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}\n")
            f.write(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}\n")
            f.write(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}\n")
            f.write(f"F1-Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}\n\n")
            
            # Classification report
            f.write("Detailed Classification Report:\n")
            f.write(classification_report(y_true, y_pred))
            f.write("\n")
            
            # Confusion matrix
            f.write("Confusion Matrix:\n")
            cm = confusion_matrix(y_true, y_pred)
            f.write(str(cm))
            f.write("\n")
        
        print(f"Evaluation report saved to: {output_path}")


def main():
    """
    Main function to run model evaluation
    """
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'best_model.pkl')
    data_path = os.path.join(project_root, 'dataset', 'processed_data.csv')
    results_dir = os.path.join(project_root, 'results')
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    print("=== Loading Data ===")
    df = pd.read_csv(data_path)
    
    # Prepare data
    from src.data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor(data_path)
    preprocessor.df = df
    preprocessor.target_column = 'Accident_severity'  # Fixed: lowercase 's'
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_data_for_modeling(
        test_size=0.2,
        random_state=42
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path)
    
    # Make predictions
    print("\n=== Making Predictions ===")
    y_pred = evaluator.predict(X_test)
    
    # Evaluate model
    evaluator.evaluate_classification(y_test, y_pred)
    
    # Print classification report
    target_names = ['Minor', 'Serious', 'Fatal']  # Adjust based on your data
    evaluator.print_classification_report(y_test, y_pred, target_names=target_names)
    
    # Plot confusion matrix
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    evaluator.plot_confusion_matrix_custom(y_test, y_pred, labels=target_names, save_path=cm_path)
    
    # Plot class distribution
    dist_path = os.path.join(results_dir, 'class_distribution.png')
    evaluator.plot_class_distribution(y_test, y_pred, labels=target_names, save_path=dist_path)
    
    # Plot feature importance
    fi_path = os.path.join(results_dir, 'feature_importance.png')
    evaluator.plot_feature_importance_custom(preprocessor.feature_columns, save_path=fi_path)
    
    # Analyze misclassifications
    misclassified = evaluator.analyze_misclassifications(
        X_test, y_test, y_pred, preprocessor.feature_columns
    )
    
    # Calculate per-class metrics
    per_class_metrics = evaluator.calculate_per_class_metrics(y_test, y_pred, target_names)
    
    # Save evaluation report
    report_path = os.path.join(results_dir, 'evaluation_report.txt')
    evaluator.save_evaluation_report(y_test, y_pred, report_path)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return evaluator


if __name__ == "__main__":
    evaluator = main()

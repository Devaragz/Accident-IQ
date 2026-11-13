"""
Utility Functions - helper functions for data processing, visualization, and model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import os


def load_dataset(file_path):
    """
    Load dataset from CSV file
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def save_dataset(df, file_path):
    """
    Save dataset to CSV file
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"Dataset saved successfully to: {file_path}")
    except Exception as e:
        print(f"Error saving dataset: {e}")


def check_missing_values(df):
    """
    Check and display missing values in the dataset
    """
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    missing_table = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    })
    missing_table = missing_table[missing_table['Missing Values'] > 0]
    missing_table = missing_table.sort_values(by='Percentage', ascending=False)  # type: ignore[call-overload]
    
    if len(missing_table) > 0:
        print("\n=== Missing Values Summary ===")
        print(missing_table)
    else:
        print("\nNo missing values found in the dataset!")
    
    return missing_table


def get_basic_info(df):
    """
    Display basic information about the dataset
    """
    print("\n=== Dataset Basic Information ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumn Names and Types:")
    print(df.dtypes)
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nBasic Statistics:")
    print(df.describe())


def plot_distribution(df, column, title=None, save_path=None):
    """
    Plot distribution of a categorical or numerical column
    """
    plt.figure(figsize=(10, 6))
    
    if df[column].dtype == 'object' or df[column].nunique() < 20:
        # Categorical or discrete
        df[column].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
        plt.xticks(rotation=45, ha='right')
    else:
        # Numerical
        df[column].hist(bins=30, color='skyblue', edgecolor='black')
    
    plt.title(title or f'Distribution of {column}', fontsize=14, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_correlation_heatmap(df, save_path=None):
    """
    Plot correlation heatmap for numerical features
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 0:
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                    center=0, fmt='.2f', linewidths=1)
        plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        
        plt.show()
    else:
        print("No numerical columns found for correlation analysis")


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Plot confusion matrix
    """
    from typing import cast, List
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    if labels is not None:
        label_list = cast(List[str], labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_list, yticklabels=label_list,
                    linewidths=1, linecolor='black')
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    linewidths=1, linecolor='black')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model and return metrics
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n=== {model_name} Performance ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print(f"\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }


def save_model(model, file_path):
    """
    Save trained model using joblib
    """
    try:
        joblib.dump(model, file_path)
        print(f"Model saved successfully to: {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model(file_path):
    """
    Load trained model using joblib
    """
    try:
        model = joblib.load(file_path)
        print(f"Model loaded successfully from: {file_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def create_directory(directory_path):
    """
    Create directory if it doesn't exist
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    """
    Plot feature importance for tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance', fontsize=14, fontweight='bold')
        plt.barh(range(top_n), importances[indices], color='skyblue', edgecolor='black')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
    else:
        print("Model does not have feature_importances_ attribute")


def compare_models(results_df, save_path=None):
    """
    Compare multiple models and visualize results
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        results_df.plot(x='Model', y=metric, kind='bar', ax=ax, 
                        color='skyblue', edgecolor='black', legend=False)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")
    
    plt.show()

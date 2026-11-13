"""
Data Preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_dataset, save_dataset, check_missing_values, get_basic_info


class DataPreprocessor:
    """
    Data Preprocessing Class for handling all data cleaning and transformation tasks
    """
    
    def __init__(self, data_path):
        """
        Initialize the preprocessor
        """
        self.data_path = data_path
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'Accident_severity'
        
    def load_data(self):
        """Load the dataset"""
        self.df = load_dataset(self.data_path)
        if self.df is not None:
            get_basic_info(self.df)
        return self.df
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset
        """
        print("\n=== Handling Missing Values ===")
        
        # Check missing values
        missing_summary = check_missing_values(self.df)
        
        if len(missing_summary) > 0:
            # For numerical columns, fill with median
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                    #self.df[col].fillna(self.df[col].median(), inplace=True)
                    print(f"Filled missing values in {col} with median")
            
            # For categorical columns, fill with mode
            categorical_cols = self.df.select_dtypes(include=['object', 'string']).columns
            #categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                    #self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    print(f"Filled missing values in {col} with mode")
        
        print("Missing value handling completed!")
        return self.df
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        print("\n=== Removing Duplicates ===")
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed_rows = initial_rows - len(self.df)
        print(f"Removed {removed_rows} duplicate rows")
        return self.df
    
    def clean_column_names(self):
        """Clean and standardize column names"""
        print("\n=== Cleaning Column Names ===")
        # Remove leading/trailing spaces and replace spaces with underscores
        self.df.columns = self.df.columns.str.strip().str.replace(' ', '_')
        # Remove special characters
        self.df.columns = self.df.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
        print("Column names cleaned successfully")
        print(f"Columns: {list(self.df.columns)}")
        return self.df
    
    def handle_outliers(self, columns=None, method='iqr', threshold=1.5):
        """
        Handle outliers in numerical columns
        """
        print("\n=== Handling Outliers ===")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                    
                    # Cap outliers instead of removing
                    self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)
                    
                    if outliers > 0:
                        print(f"Capped {outliers} outliers in {col}")
        
        print("Outlier handling completed!")
        return self.df
    
    def encode_categorical_features(self, columns=None):
        """
        Encode categorical features using Label Encoding
        """
        print("\n=== Encoding Categorical Features ===")
        
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'string']).columns
            #columns = self.df.select_dtypes(include=['object']).columns
            # Exclude target column if it's already in object type
            if self.target_column in columns:
                columns = [col for col in columns if col != self.target_column]
        
        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}: {len(le.classes_)} unique values")
        
        print("Categorical encoding completed!")
        return self.df
    
    def encode_target_variable(self):
        """
        Encode target variable (Accident_Severity)
        """
        print("\n=== Encoding Target Variable ===")
        
        if self.target_column in self.df.columns:
            le = LabelEncoder()
            self.df[self.target_column] = le.fit_transform(self.df[self.target_column].astype(str))
            self.label_encoders[self.target_column] = le
            print(f"Target classes: {le.classes_}")
            print(f"Encoded as: {list(range(len(le.classes_)))}")
        else:
            print(f"Warning: Target column '{self.target_column}' not found!")
        
        return self.df
    
    def create_features(self):
        """
        Create additional features if needed
        """
        print("\n=== Creating Additional Features ===")
        
        # Example: Create a total casualties feature if killed and injured columns exist
        if 'Persons_Killed' in self.df.columns and 'Persons_Injured' in self.df.columns:
            self.df['Total_Casualties'] = self.df['Persons_Killed'] + self.df['Persons_Injured']
            print("Created feature: Total_Casualties")
        
        # Example: Create severity indicators
        if 'Persons_Killed' in self.df.columns:
            self.df['Has_Fatality'] = (self.df['Persons_Killed'] > 0).astype(int)
            print("Created feature: Has_Fatality")
        
        print("Feature creation completed!")
        return self.df
    
    def normalize_features(self, columns=None):
        """
        Normalize numerical features using StandardScaler
        """
        print("\n=== Normalizing Features ===")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            # Exclude target column
            if self.target_column in columns:
                columns = [col for col in columns if col != self.target_column]
        
        if len(columns) > 0:
            self.df[columns] = self.scaler.fit_transform(self.df[columns])
            print(f"Normalized {len(columns)} numerical features")
        
        return self.df
    
    def prepare_data_for_modeling(self, test_size=0.2, random_state=42):
        """
        Prepare data for model training
        """
        print("\n=== Preparing Data for Modeling ===")
        
        # Separate features and target
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset!")
        
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Target distribution in training set:")
        print(y_train.value_counts())
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, save_processed=True, output_path=None):
        """
        Execute complete preprocessing pipeline
        """
        print("\n" + "="*60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Clean column names
        self.clean_column_names()
        
        # Step 3: Remove duplicates
        self.remove_duplicates()
        
        # Step 4: Handle missing values
        self.handle_missing_values()
        
        # Step 5: Create additional features
        self.create_features()
        
        # Step 6: Encode categorical features
        self.encode_categorical_features()
        
        # Step 7: Encode target variable
        self.encode_target_variable()

        # Step 8: Save preprocessed data
        if save_processed:
            if output_path is None:
                output_path = os.path.join(
                    os.path.dirname(self.data_path), 
                    'processed_data.csv'
                )
            save_dataset(self.df, output_path)
        
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return self.df


def main():
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Try different dataset filenames
    possible_files = ['Road.csv', 'road_accidents_india_2017_2022.csv']
    data_path = None
    for filename in possible_files:
        temp_path = os.path.join(project_root, 'dataset', filename)
        if os.path.exists(temp_path):
            data_path = temp_path
            break
    
    if data_path is None:
        raise FileNotFoundError("Dataset not found! Please place Road.csv in the dataset folder.")
    
    output_path = os.path.join(project_root, 'dataset', 'processed_data.csv')
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_path)
    
    # Run preprocessing pipeline
    processed_df = preprocessor.preprocess_pipeline(
        save_processed=True,
        output_path=output_path
    )
    
    print("\nPreprocessing completed! Ready for model training.")
    
    return preprocessor


if __name__ == "__main__":
    preprocessor = main()

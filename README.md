# Accident - IQ

## Road Safety Intelligence – India (2019-2024)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-red.svg)](https://xgboost.readthedocs.io/)

A comprehensive machine learning project that predicts the severity level of road accidents (Minor / Serious / Fatal) based on various environmental and situational factors using the Indian Road Accident Dataset (2019–2024).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Web Application](#web-application)
- [Technologies Used](#technologies-used)

---

## 🎯 Overview

Road accidents are a major public health concern in India, causing thousands of deaths and injuries annually. This project leverages machine learning to:

- **Predict accident severity** (Minor, Serious, Fatal)
- **Identify key risk factors** contributing to severe accidents
- **Provide actionable insights** for road safety improvements
- **Enable proactive safety measures** based on predictions

The project analyzes 6 years of accident data (2019-2024) from across India, incorporating features such as weather conditions, road type, lighting, and casualty information.

---

## ✨ Features

### Machine Learning Pipeline
- ✅ Comprehensive data preprocessing and cleaning
- ✅ Exploratory data analysis with rich visualizations
- ✅ Multiple ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- ✅ Hyperparameter tuning using GridSearchCV
- ✅ Model evaluation with multiple metrics
- ✅ Feature importance analysis

### Web Application
- ✅ Interactive Flask web app for predictions
- ✅ User-friendly interface for inputting accident details
- ✅ Real-time severity predictions
- ✅ Safety recommendations based on predicted severity
- ✅ Visualization of model performance and insights

---

## 📁 Project Structure

```
Accident-IQ/
│
├── README.md                          # Project documentation (this file)
├── requirements.txt                   # Python dependencies
├── verify_setup.py                    # Verify the setup
|
├── dataset/                           # Dataset directory
│   ├── road_accidents_india_2017_2022.csv  # Raw dataset (place here)
│   ├── processed_data.csv            # Preprocessed dataset (generated)
│   └── data_description.txt          # Dataset documentation
│
├── src/                               # Source code modules
│   ├── data_preprocessing.py         # Data preprocessing functions
│   ├── model_training.py             # Model training functions
│   ├── model_evaluation.py           # Model evaluation functions
│   ├── utils.py                      # Utility functions
│   └── predict.py                    # Prediction module
│
├── app/                               # Web application
│   ├── app.py                        # Streamlit application
│   ├── templates/                    # HTML templates
│   │   └── index.html
│   └── static/                       # CSS and static files
│       └── styles.css
│
├── results/                           # Results and outputs
│   ├── confusion_matrix.png         # Confusion matrix visualization
│   ├── feature_importance.png       # Feature importance plot
│   ├── model_performance.csv        # Model metrics comparison
│   └── severity_distribution.png    # Target variable distribution
│
└── best_model.pkl                    # Saved best performing model

```

---


## 💻 Usage

### 1. Data Exploration and Preprocessing

#### Using Python Scripts:
```bash
# Run data preprocessing
python src/data_preprocessing.py

# This will:
# - Load the raw dataset
# - Clean and preprocess the data
# - Save processed data to dataset/processed_data.csv
```

### 2. Model Training

#### Using Python Script:
```bash
python src/model_training.py

# This will:
# - Train 4 ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
# - Compare model performance
# - Save the best model to best_model.pkl
# - Save performance metrics to results/model_performance.csv
```

### 3. Model Evaluation

#### Using Python Script:
```bash
python src/model_evaluation.py

# This will:
# - Evaluate the best model
# - Generate confusion matrix
# - Create performance visualizations
# - Save results to results/ directory
```

### 4. Making Predictions

#### Console-based Prediction:
```bash
python src/predict.py

# Follow the interactive prompts to input accident details
```

#### Batch Prediction from CSV:
```python
from src.predict import predict_from_file

predict_from_file('input.csv', 'output.csv')
```

### 5. Running the Web Application

#### Flask App :
```bash
python app/app.py

# The app will open in browser at http://localhost:8501
```

---

## 📊 Dataset

### Source
- **Name**: Indian Road Accident Dataset (2017–2022)
- **Source**: Ministry of Road Transport & Highways
- **Time Period**: 2017 to 2022
- **Coverage**: Multiple states and union territories across India

### Features

| Feature | Description | Type |
|---------|-------------|------|
| Year | Year of accident (2019-2024) | Numerical |
| State/UT | State or Union Territory | Categorical |
| Road_Type | Highway, Urban, Rural | Categorical |
| Weather_Condition | Clear, Rainy, Foggy, etc. | Categorical |
| Light_Condition | Daylight, Night, etc. | Categorical |
| Road_Surface | Dry, Wet, Slippery | Categorical |
| Vehicles_Involved | Number of vehicles | Numerical |
| Persons_Killed | Number of fatalities | Numerical |
| Persons_Injured | Number of injuries | Numerical |
| **Accident_Severity** | **Minor, Serious, Fatal** | **Target** |

See `dataset/data_description.txt` for detailed feature descriptions.

---

## 🤖 Models

Four machine learning models were trained and compared:

### 1. Logistic Regression
- **Type**: Linear classification model
- **Use Case**: Baseline model for comparison
- **Pros**: Fast, interpretable
- **Cons**: Assumes linear relationships

### 2. Decision Tree
- **Type**: Tree-based classifier
- **Use Case**: Capturing non-linear patterns
- **Pros**: Interpretable, handles non-linearity
- **Cons**: Prone to overfitting

### 3. Random Forest ⭐
- **Type**: Ensemble of decision trees
- **Use Case**: High-performance classification
- **Pros**: Robust, reduces overfitting, feature importance
- **Cons**: Less interpretable, slower

### 4. XGBoost ⭐
- **Type**: Gradient boosting algorithm
- **Use Case**: State-of-the-art performance
- **Pros**: High accuracy, handles imbalanced data
- **Cons**: Requires tuning, computationally intensive

### Model Selection
The best-performing model is automatically selected based on accuracy and saved as `best_model.pkl`.

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | XX.XX% | XX.XX% | XX.XX% | XX.XX% |
| Decision Tree | XX.XX% | XX.XX% | XX.XX% | XX.XX% |
| Random Forest | XX.XX% | XX.XX% | XX.XX% | XX.XX% |
| XGBoost | XX.XX% | XX.XX% | XX.XX% | XX.XX% |

*Note: Actual metrics will be generated after training on your dataset*

### Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision and recall

### Visualizations

All visualizations are saved in the `results/` directory:
- Confusion matrix
- Feature importance
- Model comparison charts
- Severity distribution
- Class-wise performance

---

## 🌐 Web Application

### Flask App Features

1. **Interactive Prediction Interface**
   - Input fields for all accident parameters
   - Real-time severity prediction
   - Confidence scores for each severity level

2. **Insights Dashboard**
   - Dataset statistics
   - Model performance visualizations
   - Feature importance analysis

3. **Safety Recommendations**
   - Customized safety tips based on predicted severity
   - Risk mitigation strategies

### Running the App

```bash
python app/app.py
```

Navigate to `http://localhost:8501` in browser.

## 🛠️ Technologies Used

### Programming Language
- **Python 3.8+**

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning models and tools
- **xgboost** - Gradient boosting framework

### Visualization
- **matplotlib** - Static plotting
- **seaborn** - Statistical visualizations
- **plotly** - Interactive visualizations

### Web Framework
- **Flask** - Web application framework

### Development Tools
- **joblib** - Model persistence

---

## 🔄 Workflow

1. **Data Collection** → Download dataset from Ministry of Road Transport & Highways
2. **Data Exploration** → Analyze patterns and distributions
3. **Data Preprocessing** → Clean, encode, and transform data
4. **Feature Engineering** → Create new informative features
5. **Model Training** → Train multiple ML models
6. **Model Evaluation** → Compare and select best model
7. **Model Deployment** → Deploy via Streamlit web app
8. **Prediction** → Make predictions on new data

---

## 📝 Key Insights

Based on the analysis of the dataset:

1. **Weather Impact**: Rainy and foggy conditions significantly increase accident severity
2. **Lighting Conditions**: Night-time accidents tend to be more severe
3. **Road Type**: Highways have higher fatal accident rates
4. **Casualty Correlation**: Number of persons killed is a strong predictor of severity
5. **Temporal Patterns**: Certain times of year show higher accident rates

---


## 🙏 Acknowledgments

- **Data Source**: Ministry of Road Transport and Highways, Government of India
- **Inspiration**: Road safety improvement initiatives

---

## 📚 References

1. Road Safety Data - Ministry of Road Transport and Highways, India
2. Scikit-learn Documentation: https://scikit-learn.org/
3. XGBoost Documentation: https://xgboost.readthedocs.io/

---


<div align="center">

</div>

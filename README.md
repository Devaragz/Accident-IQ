# 🚗 Road Accident Severity Prediction

## Road Safety Intelligence – India (2017-2022)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-FF4B4B.svg)](https://streamlit.io/)

A comprehensive machine learning project that predicts the severity level of road accidents (Minor / Serious / Fatal) based on various environmental and situational factors using the Indian Road Accident Dataset (2017–2022).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Web Application](#web-application)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

Road accidents are a major public health concern in India, causing thousands of deaths and injuries annually. This project leverages machine learning to:

- **Predict accident severity** (Minor, Serious, Fatal)
- **Identify key risk factors** contributing to severe accidents
- **Provide actionable insights** for road safety improvements
- **Enable proactive safety measures** based on predictions

The project analyzes 6 years of accident data (2017-2022) from across India, incorporating features such as weather conditions, road type, lighting, and casualty information.

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
- ✅ Interactive Streamlit web app for predictions
- ✅ User-friendly interface for inputting accident details
- ✅ Real-time severity predictions
- ✅ Safety recommendations based on predicted severity
- ✅ Visualization of model performance and insights

### Documentation
- ✅ 5 detailed Jupyter notebooks for analysis
- ✅ Comprehensive code documentation
- ✅ Dataset description and metadata
- ✅ Setup and usage instructions

---

## 📁 Project Structure

```
Road_Accident_Severity_Prediction/
│
├── README.md                          # Project documentation (this file)
├── requirements.txt                   # Python dependencies
│
├── dataset/                           # Dataset directory
│   ├── road_accidents_india_2017_2022.csv  # Raw dataset (place here)
│   ├── processed_data.csv            # Preprocessed dataset (generated)
│   └── data_description.txt          # Dataset documentation
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Initial data exploration
│   ├── 02_data_preprocessing.ipynb   # Data cleaning and preprocessing
│   ├── 03_model_training.ipynb       # Model training and comparison
│   ├── 04_model_evaluation.ipynb     # Detailed model evaluation
│   └── 05_visualization_and_insights.ipynb  # EDA and visualizations
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

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Step-by-Step Installation

1. **Clone the repository** (or download ZIP)
   ```bash
   git clone https://github.com/yourusername/road-accident-severity-prediction.git
   cd road-accident-severity-prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download the Indian Road Accident Dataset from [Kaggle](https://www.kaggle.com/)
   - Place the CSV file (`Road.csv` or similar) in the `dataset/` directory
   - Rename it to `road_accidents_india_2017_2022.csv`

---

## 💻 Usage

### 1. Data Exploration and Preprocessing

#### Using Jupyter Notebooks:
```bash
jupyter notebook
```
Then navigate to `notebooks/` and run the notebooks in order (01, 02, 03, 04, 05).

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

#### Using Jupyter Notebook:
Open and run `notebooks/03_model_training.ipynb`

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

#### Using Jupyter Notebook:
Open and run `notebooks/04_model_evaluation.ipynb`

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

#### Streamlit App (Recommended):
```bash
streamlit run app/app.py

# The app will open in your browser at http://localhost:8501
```

---

## 📊 Dataset

### Source
- **Name**: Indian Road Accident Dataset (2017–2022)
- **Source**: Kaggle
- **Time Period**: 2017 to 2022
- **Coverage**: Multiple states and union territories across India

### Features

| Feature | Description | Type |
|---------|-------------|------|
| Year | Year of accident (2017-2022) | Numerical |
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

### Streamlit App Features

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
streamlit run app/app.py
```

Navigate to `http://localhost:8501` in your browser.

### App Screenshots

*Add screenshots of your Streamlit app here*

---

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
- **streamlit** - Web application framework

### Development Tools
- **Jupyter Notebook** - Interactive development
- **joblib** - Model persistence

---

## 🔄 Workflow

1. **Data Collection** → Download dataset from Kaggle
2. **Data Exploration** → Analyze patterns and distributions
3. **Data Preprocessing** → Clean, encode, and transform data
4. **Feature Engineering** → Create new informative features
5. **Model Training** → Train multiple ML models
6. **Model Evaluation** → Compare and select best model
7. **Hyperparameter Tuning** → Optimize model performance
8. **Model Deployment** → Deploy via Streamlit web app
9. **Prediction** → Make predictions on new data

---

## 📝 Key Insights

Based on the analysis of the dataset:

1. **Weather Impact**: Rainy and foggy conditions significantly increase accident severity
2. **Lighting Conditions**: Night-time accidents tend to be more severe
3. **Road Type**: Highways have higher fatal accident rates
4. **Casualty Correlation**: Number of persons killed is a strong predictor of severity
5. **Temporal Patterns**: Certain times of year show higher accident rates

*Detailed insights are available in the notebooks*

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Improving model performance
- Adding new features
- Enhancing visualizations
- Expanding the web application
- Writing tests
- Improving documentation

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. 

- Predictions should not be used as the sole basis for critical safety decisions
- Real-world accident severity depends on many complex factors
- Always follow official traffic safety guidelines and regulations
- Model accuracy depends on data quality and may vary

---

## 📧 Contact

**Project Maintainer**: [Your Name]
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- **Dataset**: Kaggle - Indian Road Accident Dataset
- **Data Source**: Ministry of Road Transport and Highways, Government of India
- **Inspiration**: Road safety improvement initiatives
- **Libraries**: Thanks to all open-source contributors

---

## 🗺️ Roadmap

Future enhancements planned:

- [ ] Add geospatial analysis with maps
- [ ] Implement deep learning models (Neural Networks)
- [ ] Create mobile application
- [ ] Add real-time data integration
- [ ] Expand to other countries' data
- [ ] Implement explainable AI (SHAP, LIME)
- [ ] Add API endpoints for integration
- [ ] Create automated retraining pipeline

---

## 📚 References

1. Road Safety Data - Ministry of Road Transport and Highways, India
2. Scikit-learn Documentation: https://scikit-learn.org/
3. XGBoost Documentation: https://xgboost.readthedocs.io/
4. Streamlit Documentation: https://docs.streamlit.io/

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

<div align="center">

**Made with ❤️ for Road Safety**

*"Predict. Prevent. Protect."*

</div>

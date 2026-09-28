# 💧 Hydro Forecaster

### Groundwater Level Prediction & Forecasting using Machine Learning and LSTM

Hydro Forecaster is a machine learning-based groundwater analysis and forecasting system designed to analyze groundwater-level patterns across different districts and mandals and predict future groundwater-level categories using historical groundwater and weather data.

The project combines **data preprocessing, feature engineering, supervised machine learning, and LSTM-based time-series forecasting**.

---

## 📌 Project Overview

Groundwater levels are influenced by several environmental factors such as:

- 🌧️ Rainfall
- 🌡️ Temperature
- 💧 Humidity
- 🌬️ Wind Speed
- 📅 Seasonal and temporal patterns
- 📈 Previous groundwater-level conditions

The main objective of Hydro Forecaster is to analyze these factors and predict groundwater-level categories for different geographical locations.

---

## 🎯 Objectives

- Analyze historical groundwater-level patterns.
- Study the relationship between groundwater levels and weather conditions.
- Engineer meaningful temporal and lag-based features.
- Predict groundwater-level categories using supervised ML algorithms.
- Forecast future groundwater-level categories using LSTM.
- Evaluate and compare different machine learning models.
- Generate visual insights for groundwater trends.

---

# 📊 Dataset

The primary dataset used in the project is:

Daily.csv

The dataset contains 445,212 daily observations.

Dataset Features
Column	Description
District	District name
Mandal	Mandal name
Date	Daily observation date
Rain (mm)	Daily rainfall
Avg_Temp	Average temperature
Avg_Humidity	Average humidity
Avg_Wind_Speed	Average wind speed
GW_Level_Legend	Groundwater-level category
💧 Groundwater-Level Categories

The groundwater-level category is encoded for machine learning.

Class	Groundwater Level
0	Shallow Water Level (0 to 5)
1	Moderate Water Level (5 to 10)
2	Moderately Deep Water Level (10 to 20)
3	Deep Water Level (20 to 40)
4	Very Deep Water Level (40+)

Note: These values represent categorical classes and should not be interpreted as exact groundwater-depth measurements.

🏗️ Project Architecture
                    Daily Groundwater Data
                             │
                             ▼
                     Data Preprocessing
                             │
                             ▼
                    Monthly Aggregation
                             │
                             ▼
                     Feature Engineering
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
               LSTM                 Supervised ML
                  │                     │
                  ▼                     ▼
          Time-Series             Classification
           Forecasting                Models
                  │                     │
                  └──────────┬──────────┘
                             ▼
                     Model Evaluation
                             │
                             ▼
                     Future Prediction
⚙️ Data Preprocessing

The daily groundwater dataset is transformed into a monthly time-series dataset.

Main preprocessing steps
Load Daily.csv.
Convert the Date column to datetime.
Remove unnecessary columns.
Encode groundwater-level categories.
Create a monthly time index.
Aggregate observations by:
District
Mandal
Month
Calculate monthly weather averages.
Calculate the monthly mode of groundwater-level classes.
🧠 Feature Engineering

Several additional features are created to capture temporal and historical patterns.

Weather Features
Rain (mm)
Avg_Temp
Avg_Humidity
Avg_Wind_Speed
Temporal Features
Month
Season

Seasons are classified as:

Winter
Summer
Monsoon
Post-Monsoon
Lag Features

Previous-month information is incorporated using:

GW_Level_Lag1
Rain_Lag1
Rolling Features

Three-month moving averages are calculated:

Rain_3MA
GW_Level_3MA
Location Encoding

Mandal names are converted into numerical values using:

LabelEncoder

Resulting feature:

Mandal_Code
🤖 Machine Learning Models

Two different approaches are used.

1. LSTM Time-Series Forecasting

LSTM is used to capture temporal relationships between consecutive months.

LSTM Architecture
Input Sequence
      │
      ▼
LSTM - 64 Units
      │
      ▼
Dropout - 0.2
      │
      ▼
LSTM - 32 Units
      │
      ▼
Dropout - 0.2
      │
      ▼
Dense + Softmax
      │
      ▼
Groundwater-Level Class
Configuration
Parameter	Value
Sequence Length	3 months
LSTM Layers	2
First LSTM	64 units
Second LSTM	32 units
Dropout	0.2
Optimizer	Adam
Loss	Categorical Cross-Entropy
Output	Multi-Class Softmax

The previous 3 months are used to predict the groundwater-level class of the next month.

The model also supports recursive forecasting for up to 6 future months.

🌲 2. Supervised Machine Learning

Traditional supervised classification models are also used to compare performance with the LSTM approach.

Models evaluated include:

Random Forest
XGBoost
Gradient Boosting
K-Nearest Neighbors (KNN)
Input Features
Rain (mm)
Avg_Temp
Avg_Humidity
Avg_Wind_Speed
GW_Level_Lag1
Rain_Lag1
Rain_3MA
Month
Mandal_Code
Target
GW_Level_Legend

The target is label encoded into groundwater-level classes.

📈 Model Evaluation

The models are evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Current Experimental Results
Model	Accuracy	Macro F1	Weighted F1
Random Forest	0.53	0.47	0.52
KNN (k=5)	0.56	0.47	0.53
XGBoost	0.61	0.51	0.57
Gradient Boosting	0.61	0.51	0.57

These results correspond to the current experimental setup and data split. They are not universal performance values.

🔧 Hyperparameter Tuning

Based on the current experiments, XGBoost and Gradient Boosting are candidates for further hyperparameter tuning.

For XGBoost, important parameters include:

n_estimators
max_depth
learning_rate
subsample
colsample_bytree
min_child_weight
gamma

Possible optimization techniques:

GridSearchCV
RandomizedSearchCV
Bayesian Optimization

For time-series data, chronological or time-series-aware validation should be preferred over randomly shuffling observations.

📊 Visualizations

The project generates several visualizations to understand groundwater behavior and model performance.

Exploratory Analysis
Groundwater-level trends
Rainfall trends
Temperature trends
Humidity trends
Wind-speed trends
Correlation matrix
Monthly groundwater-level distribution
Seasonal groundwater-level distribution
Rainfall vs groundwater level
Temperature vs groundwater level
Model Analysis
Confusion Matrix
Classification Report
Feature Importance
Actual vs Predicted plots
Future groundwater forecast heatmaps
🔮 Future Forecasting

The LSTM model performs recursive forecasting.

For each mandal:

Historical 3 Months
       ↓
      LSTM
       ↓
Month + 1 Prediction
       ↓
Update Sequence
       ↓
Month + 2 Prediction
       ↓
Update Sequence
       ↓
...
       ↓
Month + 6 Prediction

This allows the system to generate a 6-month groundwater-level category forecast.

💾 Model Artifacts

The trained LSTM model and preprocessing components are saved in:

model_artifacts/

Example:

model_artifacts/
│
├── gw_lstm_model.h5
├── minmax_scaler.pkl
├── mandal_label_encoder.pkl
├── features_list.pkl
└── config.pkl

Supervised ML models can also be saved using joblib.

🛠️ Technologies Used
Programming
Python
Data Processing
Pandas
NumPy
Machine Learning
Scikit-learn
XGBoost
Deep Learning
TensorFlow
Keras
LSTM
Visualization
Matplotlib
Seaborn
Plotly
Model Persistence
Joblib
Deployment
Streamlit
🚀 Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Navigate to the project:

cd GWP

Create a virtual environment:

python3.11 -m venv tf_env

Activate it:

macOS / Linux
source tf_env/bin/activate
Windows
tf_env\Scripts\activate

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost joblib streamlit
🍎 macOS XGBoost Setup

If XGBoost produces an error such as:

Library not loaded: @rpath/libomp.dylib

install OpenMP:

brew install libomp

Then run the project again.

▶️ Running the Project
Supervised ML

Run:

python supervisedmodel.py
LSTM

Run the notebook or Python script containing the LSTM preprocessing, training, evaluation, and forecasting pipeline.

📁 Project Structure
GWP/
│
├── Daily.csv
├── monthly_df.csv
│
├── hydro_forecaster.ipynb
├── supervisedmodel.py
│
├── model_artifacts/
│   ├── gw_lstm_model.h5
│   ├── minmax_scaler.pkl
│   ├── mandal_label_encoder.pkl
│   ├── features_list.pkl
│   └── config.pkl
│
├── README.md
└── requirements.txt
⚠️ Classification vs Regression

The current dataset contains groundwater-level ranges/categories, rather than exact numerical groundwater-depth measurements.

Therefore, the current implementation treats the problem as:

Groundwater Category
        ↓
Label Encoding
        ↓
Classification

For example:

0 → Shallow
1 → Moderate
2 → Moderately Deep
3 → Deep
4 → Very Deep

These values should not be treated as continuous groundwater-depth measurements.

A true groundwater regression model would require an actual numerical groundwater-depth target, for example:

Groundwater Depth = 27.4 meters

With such data, models such as:

Random Forest Regressor
XGBoost Regressor
Gradient Boosting Regressor
LSTM Regression

could be implemented.

🔮 Future Improvements

Potential improvements include:

Hyperparameter tuning using time-series cross-validation
Class balancing and class weighting
Improving prediction of minority groundwater classes
Adding historical groundwater-depth measurements
Adding water-usage/withdrawal data
Adding soil characteristics
Adding geological features
Adding elevation information
Adding geographical distance/features
SHAP-based model explainability
Automated model comparison
Streamlit dashboard deployment
FastAPI model-serving API
Continuous model retraining
👨‍💻 Author

Laxminivas Reddy Uppula

Computer Science & Engineering
Artificial Intelligence & Machine Learning

⭐ Project

Hydro Forecaster — Groundwater Level Prediction & Forecasting

A machine-learning project combining LSTM time-series forecasting and supervised machine learning classification for groundwater-level analysis.

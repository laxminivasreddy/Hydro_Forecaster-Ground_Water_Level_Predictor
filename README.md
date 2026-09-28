# Hydro Forecaster: Groundwater Level Prediction & Forecasting

## Overview

**Hydro Forecaster** is a machine learning system for analyzing
groundwater-level patterns across districts and mandals and predicting
future groundwater-level categories using weather and historical
information.

The project combines: - Data preprocessing and monthly aggregation -
Feature engineering - LSTM time-series forecasting - Supervised
machine-learning classification - Model evaluation and visualization -
Future groundwater-level forecasting

## Problem Statement

Groundwater levels are influenced by rainfall, temperature, humidity,
seasonal variation, and previous groundwater conditions. The objective
of Hydro Forecaster is to analyze these relationships and predict
groundwater-level categories for different locations.

## Dataset

The primary dataset is `Daily.csv`.

  Column              Description
  ------------------- ----------------------------
  `District`          District name
  `Mandal`            Mandal name
  `Date`              Daily observation date
  `Rain (mm)`         Daily rainfall
  `Avg_Temp`          Average temperature
  `Avg_Humidity`      Average humidity
  `Avg_Wind_Speed`    Average wind speed
  `GW_Level_Legend`   Groundwater-level category

The daily observations are aggregated into monthly observations for the
LSTM pipeline.

## Groundwater-Level Classes

The groundwater-level categories are encoded as classes:

    Class Groundwater Level
  ------- ----------------------------------------
        0 Shallow Water Level (0 to 5)
        1 Moderate Water Level (5 to 10)
        2 Moderately Deep Water Level (10 to 20)
        3 Deep Water Level (20 to 40)
        4 Very Deep Water Level (40+)

**Important:** These encoded values are categorical labels, not exact
groundwater-depth measurements.

## Project Workflow

``` text
Daily Data
   ↓
Data Cleaning
   ↓
Monthly Aggregation
   ↓
Feature Engineering
   ↓
 ┌───────────────┬──────────────────┐
 ↓               ↓
LSTM             Supervised ML
 ↓               ↓
Time-Series      Classification
Forecasting      Models
 └───────────────┴──────────────────┘
                 ↓
          Model Evaluation
                 ↓
          Future Predictions
```

## Feature Engineering

The project uses:

### Weather Features

-   Rainfall
-   Average temperature
-   Average humidity
-   Average wind speed

### Temporal Features

-   Month
-   Season

Seasons are categorized as Winter, Summer, Monsoon, and Post-Monsoon.

### Lag Features

-   `GW_Level_Lag1`
-   `Rain_Lag1`

### Rolling Features

-   `Rain_3MA`
-   `GW_Level_3MA`

### Location Feature

-   `Mandal_Code`

Mandal names are encoded using `LabelEncoder`.

## LSTM Forecasting

The LSTM model uses the previous **3 months** to predict the next
groundwater-level class.

Architecture:

``` text
Input Sequence
      ↓
LSTM (64 units)
      ↓
Dropout (0.2)
      ↓
LSTM (32 units)
      ↓
Dropout (0.2)
      ↓
Dense + Softmax
      ↓
Groundwater-Level Class
```

Configuration: - Sequence length: 3 months - Optimizer: Adam - Loss:
Categorical Cross-Entropy - Output: Multi-class Softmax

The data is split chronologically to preserve the time-series structure.

## Supervised Machine Learning

Traditional supervised classifiers are also evaluated:

-   Random Forest
-   XGBoost
-   Gradient Boosting
-   K-Nearest Neighbors (KNN)

The models use engineered tabular features and predict the
groundwater-level class.

## Model Evaluation

Models are evaluated using: - Accuracy - Precision - Recall - F1-score -
Confusion Matrix

Current experimental results:

  Model                 Accuracy   Macro F1   Weighted F1
  ------------------- ---------- ---------- -------------
  Random Forest             0.53       0.47          0.52
  KNN (k=5)                 0.56       0.47          0.53
  XGBoost                   0.61       0.51          0.57
  Gradient Boosting         0.61       0.51          0.57

These values are specific to the current experiment and data split.

## Hyperparameter Tuning

XGBoost and Gradient Boosting can be further optimized using
time-series-aware validation.

Important XGBoost parameters include:

``` text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
min_child_weight
gamma
```

Possible tuning methods: - GridSearchCV - RandomizedSearchCV - Bayesian
optimization

## Forecasting

The LSTM pipeline supports recursive forecasting for up to **6 future
months** for mandals with sufficient historical data.

Predictions can be converted back from class numbers to readable
groundwater-level categories.

## Visualizations

The project includes: - Groundwater-level trends - Rainfall,
temperature, humidity and wind trends - Correlation matrices - Monthly
and seasonal groundwater-level distributions - Rainfall vs groundwater
level - Temperature vs groundwater level - Confusion matrices - Feature
importance - Future forecast heatmaps

## Model Artifacts

The LSTM pipeline saves:

``` text
model_artifacts/
├── gw_lstm_model.h5
├── minmax_scaler.pkl
├── mandal_label_encoder.pkl
├── features_list.pkl
└── config.pkl
```

Supervised models can also be saved as `.pkl` files.

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   TensorFlow / Keras
-   XGBoost
-   Matplotlib
-   Seaborn
-   Plotly
-   Joblib
-   Streamlit

## Installation

Create a virtual environment:

``` bash
python3.11 -m venv tf_env
source tf_env/bin/activate
```

Install dependencies:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost joblib streamlit
```

### macOS XGBoost

If XGBoost reports that `libomp.dylib` cannot be loaded:

``` bash
brew install libomp
```

## Running the Project

Run the supervised classification script:

``` bash
python supervisedmodel.py
```

Run the LSTM workflow from the project notebook or Python script
containing the preprocessing, training, evaluation, and forecasting
pipeline.

## Suggested Project Structure

``` text
GWP/
├── Daily.csv
├── monthly_df.csv
├── hydro_forecaster.ipynb
├── supervisedmodel.py
├── model_artifacts/
├── README.md
└── requirements.txt
```

## Classification vs Regression

The current target is a groundwater-level **category** rather than an
exact numerical groundwater depth.

Therefore:

``` text
Label Encoding → Classification
```

is appropriate for predicting categories such as Shallow, Moderate,
Deep, and Very Deep.

A true regression model would require actual numerical groundwater-depth
measurements, for example:

``` text
Groundwater Depth = 27.4 meters
```

If exact groundwater measurements become available, models such as
Random Forest Regressor, XGBoost Regressor, Gradient Boosting Regressor,
or LSTM regression can be developed.

## Future Improvements

-   Hyperparameter tuning using time-series cross-validation
-   Class weighting or resampling for minority classes
-   Improve prediction of the shallow-water class
-   Add historical groundwater measurements
-   Add water-usage data
-   Add soil and geological features
-   Add elevation and geographical features
-   SHAP-based explainability
-   Automated model comparison
-   Streamlit deployment
-   FastAPI inference API
-   Continuous model retraining

## Deployed Link :https://hydroforecaster-kdkbrmdhszklu6lvdxc7vt.streamlit.app/

## Author

**Laxminivas Reddy Uppula**

Computer Science and Engineering -- Artificial Intelligence & Machine
Learning

Hydro Forecaster was developed as a machine-learning project for
groundwater-level analysis, classification, and time-series forecasting.

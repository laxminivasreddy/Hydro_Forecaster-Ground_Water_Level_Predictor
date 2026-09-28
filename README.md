Hydro Forecaster
Groundwater Level Prediction & Forecasting using Machine Learning and LSTM
Hydro Forecaster is a machine learning-based groundwater analysis and forecasting system designed to analyze
groundwater-level patterns across different districts and mandals and predict future groundwater-level categories using
historical groundwater and weather data.
n Objectives
• Analyze historical groundwater-level patterns.
• Study the relationship between groundwater levels and weather conditions.
• Engineer temporal, lag, and rolling features.
• Predict groundwater-level categories using supervised machine learning.
• Forecast future groundwater-level categories using LSTM.
• Compare different machine learning models.
• Generate visual insights for groundwater trends and model performance.
n Dataset
The primary dataset is Daily.csv and contains 445,212 daily observations.
Column Description
District District name
Mandal Mandal name
Date Daily observation date
Rain (mm) Daily rainfall
Avg_Temp Average temperature
Avg_Humidity Average humidity
Avg_Wind_Speed Average wind speed
GW_Level_Legend Groundwater-level category
n Groundwater-Level Categories
Class Groundwater Level
0 Shallow Water Level (0 to 5)
1 Moderate Water Level (5 to 10)
2 Moderately Deep Water Level (10 to 20)
3 Deep Water Level (20 to 40)
4 Very Deep Water Level (40+)
Note: These encoded values represent categorical classes and should not be interpreted as exact groundwater-depth
measurements.
nn Project Architecture
Daily Groundwater Data
↓
Data Preprocessing
↓
Monthly Aggregation
↓
Feature Engineering
↓
nnnnnnnnnnnnnnn
↓ ↓
LSTM Supervised ML
↓ ↓
Time-Series Classification
Forecasting Models
nnnnnnnnnnnnnnn
↓
Model Evaluation
↓
Future Prediction
nn Data Preprocessing
• Load Daily.csv.
• Convert the Date column to datetime.
• Remove unnecessary columns.
• Encode groundwater-level categories.
• Create a monthly time index.
• Aggregate observations by District, Mandal, and Month.
• Calculate monthly weather averages.
• Calculate the monthly mode of groundwater-level classes.
n Feature Engineering
Weather: Rain (mm), Avg_Temp, Avg_Humidity, Avg_Wind_Speed
Temporal: Month, Season
Lag: GW_Level_Lag1, Rain_Lag1
Rolling: Rain_3MA, GW_Level_3MA
Location: Mandal_Code using LabelEncoder
n Machine Learning Models
The project uses two major approaches: LSTM time-series forecasting and supervised machine-learning classification.
n LSTM Time-Series Forecasting
The LSTM model captures temporal relationships between consecutive monthly observations. The previous 3 months
are used to predict the next groundwater-level class.
Input Sequence
↓
LSTM - 64 Units
↓
Dropout - 0.2
↓
LSTM - 32 Units
↓
Dropout - 0.2
↓
Dense + Softmax
↓
Groundwater-Level Class
Parameter Value
Sequence Length 3 months
LSTM Layers 2
First LSTM Layer 64 units
Second LSTM Layer 32 units
Dropout 0.2
Optimizer Adam
Loss Categorical Cross-Entropy
Output Multi-Class Softmax
The LSTM supports recursive forecasting for up to 6 future months.
n Supervised Machine Learning
Traditional supervised classifiers are used to compare performance with the LSTM approach.
• Random Forest
• XGBoost
• Gradient Boosting
• K-Nearest Neighbors (KNN)
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
n Model Evaluation
Models are evaluated using accuracy, precision, recall, F1-score, and confusion matrices.
Model Accuracy Macro F1 Weighted F1
Random Forest 0.53 0.47 0.52
KNN (k=5) 0.56 0.47 0.53
XGBoost 0.61 0.51 0.57
Gradient Boosting 0.61 0.51 0.57
These results correspond to the current experimental setup and data split.
n Hyperparameter Tuning
Based on the current experiments, XGBoost and Gradient Boosting are candidates for further hyperparameter tuning.
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
min_child_weight
gamma
Possible optimization techniques include GridSearchCV, RandomizedSearchCV, and Bayesian Optimization. For
time-series data, chronological or time-series-aware validation should be preferred.
n Visualizations
• Groundwater-level trends
• Rainfall, temperature, humidity and wind trends
• Correlation matrix
• Monthly groundwater-level distribution
• Seasonal groundwater-level distribution
• Rainfall vs groundwater level
• Temperature vs groundwater level
• Confusion matrices
• Feature importance
• Actual vs predicted plots
• Future forecast heatmaps
n Future Forecasting
The LSTM model performs recursive forecasting for future groundwater-level categories. For each mandal, predictions
are generated sequentially for up to 6 future months.
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
...
↓
Month + 6 Prediction
n Model Artifacts
model_artifacts/
nnn gw_lstm_model.h5
nnn minmax_scaler.pkl
nnn mandal_label_encoder.pkl
nnn features_list.pkl
nnn config.pkl
nn Technologies Used
• Programming: Python
• Data Processing: Pandas, NumPy
• Machine Learning: Scikit-learn, XGBoost
• Deep Learning: TensorFlow, Keras, LSTM
• Visualization: Matplotlib, Seaborn, Plotly
• Model Persistence: Joblib
• Deployment: Streamlit
n Installation
python3.11 -m venv tf_env
source tf_env/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost joblib streamlit
n macOS XGBoost Setup
If XGBoost reports that libomp.dylib cannot be loaded, install OpenMP using Homebrew:
brew install libomp
nn Running the Project
Supervised Machine Learning:
python supervisedmodel.py
For LSTM forecasting, run the notebook or Python script containing the preprocessing, training, evaluation, and
forecasting pipeline.
n Project Structure
GWP/
nnn Daily.csv
nnn monthly_df.csv
nnn hydro_forecaster.ipynb
nnn supervisedmodel.py
nnn model_artifacts/
nnn README.md
nnn requirements.txt
nn Classification vs Regression
The current dataset contains groundwater-level categories/ranges rather than exact numerical groundwater-depth
measurements. Therefore, the current implementation treats the problem as classification.
Groundwater Category
↓
Label Encoding
↓
Classification
0 → Shallow
1 → Moderate
2 → Moderately Deep
3 → Deep
4 → Very Deep
These encoded values should not be treated as continuous groundwater-depth measurements. A true groundwater
regression model would require an actual numerical groundwater-depth target, such as 27.4 meters.
n Future Improvements
• Hyperparameter tuning using time-series cross-validation
• Class weighting and class balancing
• Improve prediction of minority groundwater classes
• Add historical groundwater-depth measurements
• Add water-usage/withdrawal data
• Add soil characteristics
• Add geological features
• Add elevation and geographical features
• SHAP-based model explainability
• Automated model comparison
• Streamlit dashboard deployment
• FastAPI model-serving API
• Continuous model retraining
nnn Author
Laxminivas Reddy Uppula
Computer Science & Engineering – Artificial Intelligence & Machine Learning
Hydro Forecaster — Groundwater Level Prediction & Forecasting

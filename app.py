import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import folium
from streamlit_folium import st_folium

# ========== Load Models and Artifacts ==========
# LSTM artifacts
lstm_model = load_model("gw_lstm_model.h5")
lstm_scaler = joblib.load("minmax_scaler.pkl")
lstm_label_encoder = joblib.load("mandal_label_encoder.pkl")
lstm_features = joblib.load("features_list.pkl")
lstm_config = joblib.load("config.pkl")

# XGBoost artifacts
xgb_model = joblib.load("xgboost_model.pkl")
xgb_scaler = joblib.load("scaler.pkl")
xgb_label_encoder = joblib.load("label_encoder.pkl")

# Data
df = pd.read_csv("monthly_df.csv")

# ========== Streamlit Interface ==========
st.title("💧 Groundwater Level Forecasting App")

# Select model
model_choice = st.radio("Choose a model:", ("LSTM (Forecast)", "XGBoost (Manual Input)"))

# ========== LSTM MODE ==========
# ========== LSTM MODE ==========
if model_choice == "LSTM (Forecast)":
    st.markdown("Forecast groundwater level for next 6 months using historical data")

    st.subheader("🗺️ Select Mandal from Map of Telangana")

    # Load GeoJSON
    geojson = "telangana_mandals.geojson"  # Adjust path
    m = folium.Map(location=[17.9784, 79.5941], zoom_start=7)

    # Define callback to store selected mandal
    selected_mandal = None

    def style_function(feature):
        return {
            'fillColor': 'blue',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.2
        }

    def highlight_function(feature):
        return {
            'fillColor': 'yellow',
            'color': 'black',
            'fillOpacity': 0.7,
            'weight': 2
        }

    # Add mandals to the map
    folium.GeoJson(
        geojson,
        name="Mandal Boundaries",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(fields=["DM_N"], aliases=["Mandal:"]),
    ).add_to(m)

    # Render map and capture interaction
    map_data = st_folium(m, width=700, height=500)

    # Determine selected mandal
    if map_data and map_data.get("last_active_drawing"):
        selected_mandal = map_data["last_active_drawing"]["properties"]["DM_N"]

    # Fallback: dropdown in case map doesn't load
    if not selected_mandal:
        mandals = sorted(df['Mandal'].unique())
        selected_mandal = st.selectbox("Or select a Mandal manually", mandals)

    st.success(f"📍 Selected Mandal: {selected_mandal}")

    # Filter and plot
    mandal_df = df[df['Mandal'] == selected_mandal].sort_values(by="YearMonth")
    st.subheader(f"📈 Historical Groundwater Trend for {selected_mandal}")
    st.line_chart(mandal_df[['YearMonth', 'GW_Level_Code']].set_index("YearMonth"))

    # Check enough data
    timesteps = lstm_config["sequence_length"]
    if len(mandal_df) < timesteps:
        st.warning("Not enough historical data to forecast.")
        st.stop()

    # Prepare sequence
    latest_seq = mandal_df[lstm_features].values[-timesteps:]
    scaled_seq = lstm_scaler.transform(latest_seq)

    # Forecast logic
    def forecast_lstm(model, current_seq, steps):
        preds = []
        seq = current_seq.copy()
        for _ in range(steps):
            pred = model.predict(seq.reshape(1, seq.shape[0], seq.shape[1]), verbose=0)
            pred_class = np.argmax(pred, axis=1)[0]
            preds.append(pred_class)

            new_input = seq[-1].copy()
            new_input[3] = pred_class  # Update GW_Level_Lag1
            seq = np.vstack([seq[1:], new_input])
        return preds

    future_preds = forecast_lstm(lstm_model, scaled_seq, steps=6)

    # Display heatmap
    st.subheader("🔮 Forecast: Next 6 Months")
    forecast_df = pd.DataFrame([future_preds], columns=[f"Month+{i+1}" for i in range(6)])
    forecast_df.index = [selected_mandal]

    fig, ax = plt.subplots(figsize=(10, 2))
    sns.heatmap(forecast_df, annot=True, cmap="YlGnBu", cbar=False, fmt="d", ax=ax)
    plt.xlabel("Future Months")
    plt.ylabel("Mandal")
    st.pyplot(fig)

    st.caption("LSTM Forecast | Codes: 0=Shallow, 1=Moderate, ..., 4=Very Deep")

# ========== XGBOOST MODE ==========
elif model_choice == "XGBoost (Manual Input)":
    st.markdown("Predict groundwater level using current weather inputs (no Mandal selection needed)")

    # Input form
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, step=1.0)
    temperature = st.number_input("🌡️ Average Temperature (°C)", min_value=0.0, step=0.1)
    humidity = st.number_input("💦 Average Humidity (%)", min_value=0.0, max_value=100.0, step=1.0)

    if st.button("🔍 Predict using XGBoost"):
        # Prepare input
        input_data = pd.DataFrame([{
            "Rain (mm)": rainfall,
            "Avg_Temp": temperature,
            "Avg_Humidity": humidity
        }])

        scaled_input = xgb_scaler.transform(input_data)
        pred_class = xgb_model.predict(scaled_input)[0]
        pred_label = xgb_label_encoder.inverse_transform([pred_class])[0]

        st.success(f"📊 Predicted Groundwater Level Class: **{pred_label}**")

# ========== Footer ==========
st.markdown("---")
st.caption("Built with 💡 LSTM & XGBoost | Groundwater Classification")

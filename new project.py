import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# Load Artifacts
# ---------------------------
MODEL_PATH = "best_model.joblib"
IMPUTER_PATH = "num_imputer.joblib"
SCALER_PATH = "scaler.joblib"
ENCODERS_PATH = "label_encoders.joblib"
TARGET_ENCODER_PATH = "target_encoder.joblib"

model = joblib.load(MODEL_PATH)
num_imputer = joblib.load(IMPUTER_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
target_encoder = joblib.load(TARGET_ENCODER_PATH)

# Streamlit Title
st.title("🌲 EcoType: Forest Cover Type Prediction")
st.write("Predict forest cover type using Machine Learning and Cartographic Variables")

# ---------------------------
# User Input Form
# ---------------------------

st.header("🔧 Input Features")

# Define input fields
elevation = st.number_input("Elevation (meters)", 0, 5000, 2500)
aspect = st.number_input("Aspect (degrees)", 0, 360, 90)
slope = st.number_input("Slope (degrees)", 0, 90, 10)

h_dist_hydro = st.number_input("Horizontal Distance to Hydrology", 0, 10000, 200)
v_dist_hydro = st.number_input("Vertical Distance to Hydrology", -1000, 10000, 50)
h_dist_road = st.number_input("Horizontal Distance to Roadways", 0, 10000, 500)
h_dist_fire = st.number_input("Horizontal Distance to Fire Points", 0, 10000, 1000)

hillshade_9am = st.number_input("Hillshade 9AM", 0, 255, 200)
hillshade_noon = st.number_input("Hillshade Noon", 0, 255, 220)
hillshade_3pm = st.number_input("Hillshade 3PM", 0, 255, 180)

# Wilderness inputs
wilderness_options = ["Wilderness_Area_1", "Wilderness_Area_2", "Wilderness_Area_3", "Wilderness_Area_4"]
wilderness = st.selectbox("Wilderness Area", wilderness_options)

# Soil type inputs
soil_options = [f"Soil_Type_{i}" for i in range(1, 41)]
soil = st.selectbox("Soil Type", soil_options)

# ---------------------------
# Prepare input for model
# ---------------------------

if st.button("Predict Cover Type"):

    # Build input dictionary
    input_data = {
        "Elevation": elevation,
        "Aspect": aspect,
        "Slope": slope,
        "Horizontal_Distance_To_Hydrology": h_dist_hydro,
        "Vertical_Distance_To_Hydrology": v_dist_hydro,
        "Horizontal_Distance_To_Roadways": h_dist_road,
        "Hillshade_9am": hillshade_9am,
        "Hillshade_Noon": hillshade_noon,
        "Hillshade_3pm": hillshade_3pm,
        "Horizontal_Distance_To_Fire_Points": h_dist_fire,
        wilderness: 1,
    }

    # Add other wilderness = 0
    for w in wilderness_options:
        if w not in input_data:
            input_data[w] = 0

    # Add soil type = 1
    input_data[soil] = 1

    # Add other soil types = 0
    for s in soil_options:
        if s not in input_data:
            input_data[s] = 0

    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])

    # ---------------------------
    # Preprocessing
    # ---------------------------
    num_cols = df_input.select_dtypes(include=[np.number]).columns

    # Impute numeric
    df_input[num_cols] = num_imputer.transform(df_input[num_cols])

    # Scale
    df_input[num_cols] = scaler.transform(df_input[num_cols])

    # Predict
    pred = model.predict(df_input)
    predicted_class = target_encoder.inverse_transform(pred)[0]

    st.success(f"🌳 **Predicted Forest Cover Type: {predicted_class}**")

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("StreamlitApp")

# Absolute paths for the model and scaler
model_path = r"C:\Users\sahil\Desktop\Mycotoxin Prediction\src\models\stacked_model.pkl"
scaler_path = r"C:\Users\sahil\Desktop\Mycotoxin Prediction\src\models\scaler.pkl"

# Load trained model
@st.cache_resource
def load_model():
    try:
        if os.path.exists(model_path):
            with open(model_path, "rb") as model_file:
                model = pickle.load(model_file)
            logger.info("Model loaded successfully.")
            return model
        else:
            logger.error(f"Model file not found at {model_path}")
            st.error(f"Model file not found at {model_path}. Please check the logs.")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error("Error loading the model. Please check logs.")
        return None

# Load scaler (if used)
@st.cache_resource
def load_scaler():
    try:
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as scaler_file:
                scaler = pickle.load(scaler_file)
            logger.info("Scaler loaded successfully.")
            return scaler
        else:
            logger.warning(f"Scaler file not found at {scaler_path}. Proceeding without scaling.")
            return None
    except Exception as e:
        logger.warning(f"Scaler not found. Proceeding without scaling. Error: {e}")
        return None

model = load_model()
scaler = load_scaler()

# Define the 24 selected features
selected_features = [
    '25', '35', '59', '73', '78', '84', '85', '86', '97', '123', '136', '158', '160', 
    '162', '164', '170', '182', '363', '371', '382', '384', '405', '443'
]

# Streamlit UI
st.title("🌽 Mycotoxin Prediction in Corn")
st.write("Upload spectral data or enter manually to predict DON concentration.")

# File Upload Option
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
else:
    st.write("Enter spectral data manually:")
    input_data = [st.number_input(f"Feature {feature}", value=0.0) for feature in selected_features]
    input_df = pd.DataFrame([input_data], columns=selected_features)

# Preprocess input
if scaler is not None:  # Check if scaler is loaded
    input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

# Predict Button
if st.button("Predict"):
    if model is not None:  # Check if model is loaded successfully
        prediction = model.predict(input_df)
        st.success(f"Predicted DON concentration: {prediction[0]:.4f} ppm")
        logger.info(f"Prediction made: {prediction[0]:.4f}")
    else:
        st.error("Model not available.")

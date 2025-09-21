import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Manufacturing Output Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
# This function loads your trained model from the .pkl file
@st.cache_resource
def load_model():
    """Loads the trained model from the pickle file."""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Error: model.pkl file not found. Make sure it's in your GitHub repository.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None

model = load_model()

# --- Main App ---
st.title("🏭 Manufacturing Output Predictor")

if model is None:
    st.info("Please wait while the model is loading or check the error message above.")
    st.stop()

# --- Sidebar and Input Fields ---
with st.sidebar:
    st.header("⚙️ Machine Parameters")
    injection_temp = st.slider("Injection Temperature (°C)", 180.0, 250.0, 210.0)
    injection_pressure = st.slider("Injection Pressure (bar)", 80.0, 150.0, 120.0)
    cycle_time = st.slider("Cycle Time (s)", 15.0, 60.0, 30.0)
    cooling_time = st.slider("Cooling Time (s)", 8.0, 20.0, 12.0)
    material_viscosity = st.slider("Material Viscosity (Pa·s)", 100.0, 400.0, 250.0)
    ambient_temp = st.slider("Ambient Temperature (°C)", 18.0, 30.0, 24.0)
    machine_age = st.slider("Machine Age (years)", 1.0, 15.0, 5.0)
    operator_exp = st.slider("Operator Experience (months)", 1.0, 120.0, 24.0)
    maintenance_hrs = st.slider("Maintenance Hours", 30.0, 80.0, 50.0)
    shift = st.selectbox("Shift", ["Day", "Night", "Evening"])
    machine_type = st.selectbox("Machine Type", ["Type_A", "Type_B", "Type_C"])
    material_grade = st.selectbox("Material Grade", ["Economy", "Standard", "Premium"])
    day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    # Calculate derived features automatically
    temp_pressure_ratio = injection_temp / injection_pressure if injection_pressure != 0 else 0
    total_cycle_time = cycle_time + cooling_time

    efficiency_score = st.slider("Efficiency Score", 0.01, 1.0, 0.15)
    machine_utilization = st.slider("Machine Utilization", 0.1, 1.0, 0.5)

# --- Prediction Logic ---
# Create a DataFrame from the inputs
input_data = pd.DataFrame({
    "Injection_Temperature": [injection_temp],
    "Injection_Pressure": [injection_pressure],
    "Cycle_Time": [cycle_time],
    "Cooling_Time": [cooling_time],
    "Material_Viscosity": [material_viscosity],
    "Ambient_Temperature": [ambient_temp],
    "Machine_Age": [machine_age],
    "Operator_Experience": [operator_exp],
    "Maintenance_Hours": [maintenance_hrs],
    "Shift": [shift],
    "Machine_Type": [machine_type],
    "Material_Grade": [material_grade],
    "Day_of_Week": [day_of_week],
    "Temperature_Pressure_Ratio": [temp_pressure_ratio],
    "Total_Cycle_Time": [total_cycle_time],
    "Efficiency_Score": [efficiency_score],
    "Machine_Utilization": [machine_utilization]
})

st.header("🔮 Prediction")

# Make prediction and display it
try:
    prediction = model.predict(input_data)
    st.metric(
        label="Predicted Parts Per Hour",
        value=f"{prediction[0]:.0f}"
    )
    st.info("Adjust the parameters in the sidebar to see the predicted output change in real-time.")

except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
    st.warning("""
        This may be due to a version mismatch between the model file (`model.pkl`) 
        and the `scikit-learn` library. You may need to retrain your model with the
        same library versions listed in your `requirements.txt` file and re-upload it.
    """)

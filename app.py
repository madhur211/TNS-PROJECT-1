import streamlit as st
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Manufacturing Output Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_URL = "http://localhost:8000"

def check_api_health():
    """Checks if the FastAPI server is running."""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def get_model_info():
    """Retrieves model information from the API."""
    try:
        response = requests.get(f"{API_URL}/model-info")
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None

def make_prediction(data):
    """Sends a single prediction request to the API."""
    try:
        response = requests.post(f"{API_URL}/predict", json=data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API Error: Could not get a valid prediction. Details: {e}")
        return None

def main():
    """Main function to run the Streamlit app."""
    st.markdown('<h1 class="main-header">🏭 Manufacturing Output Predictor</h1>', unsafe_allow_html=True)

    # Check API health
    if not check_api_health():
        st.error("⚠️ API server is not running. Please start the FastAPI server first.")
        st.info("In your terminal, run: uvicorn main:app --reload")
        return

    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        model_info = get_model_info()
        if model_info:
            st.write(f"**Model Type:** {model_info.get('model_type', 'N/A')}")
            st.write(f"**Training Date:** {model_info.get('training_date', 'N/A')}")
            st.write(f"**R² Score:** {model_info.get('performance_metrics', {}).get('r2', 0):.3f}")
            st.write(f"**RMSE:** {model_info.get('performance_metrics', {}).get('rmse', 0):.2f}")
        else:
            st.warning("Could not retrieve model information.")

        st.header("🔧 Navigation")
        page = st.radio("Choose a page", ["Single Prediction", "Batch Prediction", "Data Analysis"])

    # Page routing
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "Data Analysis":
        data_analysis_page()

def single_prediction_page():
    """Displays the page for making a single prediction."""
    st.header("🔍 Single Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Machine Parameters")
        injection_temp = st.slider("Injection Temperature (°C)", 180.0, 250.0, 210.0)
        injection_pressure = st.slider("Injection Pressure (bar)", 80.0, 150.0, 120.0)
        cycle_time = st.slider("Cycle Time (s)", 15.0, 60.0, 30.0)
        cooling_time = st.slider("Cooling Time (s)", 8.0, 20.0, 12.0)
        material_viscosity = st.slider("Material Viscosity (Pa·s)", 100.0, 400.0, 250.0)
        ambient_temp = st.slider("Ambient Temperature (°C)", 18.0, 30.0, 24.0)

    with col2:
        st.subheader("Additional Parameters")
        machine_age = st.slider("Machine Age (years)", 1.0, 15.0, 5.0)
        operator_exp = st.slider("Operator Experience (months)", 1.0, 120.0, 24.0)
        maintenance_hrs = st.slider("Maintenance Hours", 30.0, 80.0, 50.0)
        shift = st.selectbox("Shift", ["Day", "Night", "Evening"])
        machine_type = st.selectbox("Machine Type", ["Type_A", "Type_B", "Type_C"])
        material_grade = st.selectbox("Material Grade", ["Economy", "Standard", "Premium"])
        day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        temp_pressure_ratio = st.slider("Temperature-Pressure Ratio", 1.3, 2.8, 1.8)
        total_cycle_time = st.slider("Total Cycle Time (s)", 25.0, 65.0, 42.0)
        efficiency_score = st.slider("Efficiency Score", 0.01, 0.8, 0.15)
        machine_utilization = st.slider("Machine Utilization", 0.1, 0.8, 0.5)

    # Prepare data for prediction
    input_data = {
        "Injection_Temperature": injection_temp, "Injection_Pressure": injection_pressure,
        "Cycle_Time": cycle_time, "Cooling_Time": cooling_time,
        "Material_Viscosity": material_viscosity, "Ambient_Temperature": ambient_temp,
        "Machine_Age": machine_age, "Operator_Experience": operator_exp,
        "Maintenance_Hours": maintenance_hrs, "Shift": shift,
        "Machine_Type": machine_type, "Material_Grade": material_grade,
        "Day_of_Week": day_of_week, "Temperature_Pressure_Ratio": temp_pressure_ratio,
        "Total_Cycle_Time": total_cycle_time, "Efficiency_Score": efficiency_score,
        "Machine_Utilization": machine_utilization
    }

    # Prediction button
    if st.button("🚀 Predict Output", use_container_width=True):
        with st.spinner("Calculating..."):
            result = make_prediction(input_data)

            if result:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                # --- FIX APPLIED HERE ---
                # Use .get() to safely access keys. If 'prediction' key doesn't exist,
                # it will use the result itself. This prevents the KeyError.
                prediction_value = result.get('prediction', result)
                confidence_value = result.get('confidence', 0) # Default to 0 if not found

                st.metric(
                    label="Predicted Parts Per Hour",
                    value=f"{prediction_value:.0f}",
                    delta=f"{confidence_value:.1f}% confidence" if confidence_value else ""
                )
                st.markdown('</div>', unsafe_allow_html=True)

def batch_prediction_page():
    """Displays the page for making batch predictions from a CSV file."""
    st.header("📊 Batch Prediction")
    st.info("Upload a CSV file with the required columns for batch predictions.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(df.head())

            required_cols = [
                'Injection_Temperature', 'Injection_Pressure', 'Cycle_Time',
                'Cooling_Time', 'Material_Viscosity', 'Ambient_Temperature',
                'Machine_Age', 'Operator_Experience', 'Maintenance_Hours',
                'Shift', 'Machine_Type', 'Material_Grade', 'Day_of_Week',
                'Temperature_Pressure_Ratio', 'Total_Cycle_Time',
                'Efficiency_Score', 'Machine_Utilization'
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button("🔮 Make Batch Predictions", use_container_width=True):
                    with st.spinner("Processing..."):
                        data = df.to_dict('records')
                        try:
                            response = requests.post(f"{API_URL}/batch-predict", json=data)
                            response.raise_for_status()
                            results = response.json()

                            st.success(f"Successfully made {results.get('total_predictions', 0)} predictions!")
                            
                            # --- FIX APPLIED HERE ---
                            # Safely create a DataFrame from the prediction results
                            predictions_list = results.get('predictions', [])
                            if predictions_list:
                                results_df = pd.DataFrame(predictions_list)
                                # Combine original data with new prediction columns that exist
                                for col in ['prediction', 'confidence']:
                                    if col in results_df.columns:
                                        df[col] = results_df[col]

                            st.write("Prediction Results:")
                            st.dataframe(df)

                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )
                        except requests.RequestException as e:
                            st.error(f"Error during batch prediction: {e}")

        except Exception as e:
            st.error(f"Error reading or processing file: {e}")

def data_analysis_page():
    """Displays a simple data analysis page."""
    st.header("📈 Data Analysis & Optimization")
    st.info("This section provides insights based on the model's features.")
    
    st.subheader("Optimization Tips")
    tips = [
        "📊 **Temperature:** Maintain Injection Temperature between 210-230°C for optimal output.",
        "⚡ **Cycle Time:** Shorter cycle times (below 35s) generally lead to higher throughput.",
        "🔧 **Maintenance:** Consistent and regular maintenance significantly improves machine efficiency and lifespan.",
        "👨‍💼 **Operator Experience:** Experienced operators can increase output by up to 15%. Invest in training.",
        "🌡️ **Environment:** A stable ambient temperature improves material consistency and final part quality.",
        "🔄 **Utilization:** Aim for a machine utilization rate between 60-70% to balance output and machine health."
    ]
    for tip in tips:
        st.markdown(f"- {tip}")

if __name__ == "__main__":
    main()
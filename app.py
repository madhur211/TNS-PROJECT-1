import streamlit as st
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

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

# API configuration - Use environment variable for deployment
API_URL = os.environ.get("FASTAPI_URL", "http://localhost:8000")

def check_api_health():
    """Checks if the FastAPI server is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False

def get_model_info():
    """Retrieves model information from the API."""
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None

def make_prediction(data):
    """Sends a single prediction request to the API."""
    try:
        response = requests.post(f"{API_URL}/predict", json=data, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API Error: Could not get a valid prediction. Details: {str(e)}")
        return None

def main():
    """Main function to run the Streamlit app."""
    st.markdown('<h1 class="main-header">🏭 Manufacturing Output Predictor</h1>', unsafe_allow_html=True)

    # Check API health
    if not check_api_health():
        st.error("⚠️ API server is not reachable. Please ensure your FastAPI backend is deployed and running.")
        st.info("For local development: Run `uvicorn main:app --reload`")
        st.info("For production: Deploy your FastAPI backend and set the FASTAPI_URL environment variable")
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
        "Injection_Temperature": injection_temp,
        "Injection_Pressure": injection_pressure,
        "Cycle_Time": cycle_time,
        "Cooling_Time": cooling_time,
        "Material_Viscosity": material_viscosity,
        "Ambient_Temperature": ambient_temp,
        "Machine_Age": machine_age,
        "Operator_Experience": operator_exp,
        "Maintenance_Hours": maintenance_hrs,
        "Shift": shift,
        "Machine_Type": machine_type,
        "Material_Grade": material_grade,
        "Day_of_Week": day_of_week,
        "Temperature_Pressure_Ratio": temp_pressure_ratio,
        "Total_Cycle_Time": total_cycle_time,
        "Efficiency_Score": efficiency_score,
        "Machine_Utilization": machine_utilization
    }

    # Prediction button
    if st.button("🚀 Predict Output", use_container_width=True):
        with st.spinner("Calculating prediction..."):
            result = make_prediction(input_data)

            if result:
                # Handle different response formats safely
                if isinstance(result, dict):
                    prediction_value = result.get('prediction')
                    confidence_value = result.get('confidence', 80.0)
                else:
                    # If result is not a dictionary (e.g., direct number)
                    prediction_value = result
                    confidence_value = 80.0

                if prediction_value is not None:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.metric(
                        label="Predicted Parts Per Hour",
                        value=f"{float(prediction_value):.0f}",
                        delta=f"{float(confidence_value):.1f}% confidence"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Show additional insights
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Efficiency Score", f"{efficiency_score:.3f}")
                    with col2:
                        st.metric("Machine Utilization", f"{machine_utilization:.1%}")
                    with col3:
                        st.metric("Cycle Time", f"{cycle_time:.1f}s")
                else:
                    st.error("Prediction failed. Please check your API configuration.")
            else:
                st.error("Failed to get prediction. Please ensure your FastAPI backend is properly deployed.")

def batch_prediction_page():
    """Displays the page for making batch predictions from a CSV file."""
    st.header("📊 Batch Prediction")
    st.info("Upload a CSV file with manufacturing data for batch predictions.")

    # Show sample data format
    with st.expander("📋 Expected CSV Format"):
        st.write("""
        Your CSV should contain these columns:
        - Injection_Temperature, Injection_Pressure, Cycle_Time, Cooling_Time
        - Material_Viscosity, Ambient_Temperature, Machine_Age
        - Operator_Experience, Maintenance_Hours, Shift
        - Machine_Type, Material_Grade, Day_of_Week
        - Temperature_Pressure_Ratio, Total_Cycle_Time
        - Efficiency_Score, Machine_Utilization
        """)

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
                    with st.spinner("Processing batch predictions..."):
                        try:
                            # Convert to list of dictionaries
                            data = df[required_cols].to_dict('records')
                            
                            response = requests.post(f"{API_URL}/batch-predict", json=data, timeout=30)
                            response.raise_for_status()
                            results = response.json()

                            predictions = results.get('predictions', [])
                            if predictions:
                                # Add predictions to dataframe
                                predictions_df = pd.DataFrame(predictions)
                                result_df = df.copy()
                                if 'prediction' in predictions_df.columns:
                                    result_df['Prediction'] = predictions_df['prediction']
                                if 'confidence' in predictions_df.columns:
                                    result_df['Confidence'] = predictions_df['confidence']

                                st.success(f"✅ Successfully made {len(predictions)} predictions!")
                                st.dataframe(result_df)

                                # Download button
                                csv = result_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Predictions",
                                    data=csv,
                                    file_name="manufacturing_predictions.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("No predictions were returned from the API.")

                        except requests.RequestException as e:
                            st.error(f"API Error: {str(e)}")
                        except Exception as e:
                            st.error(f"Processing error: {str(e)}")

        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

def data_analysis_page():
    """Displays a data analysis and optimization tips page."""
    st.header("📈 Data Analysis & Optimization")
    
    st.info("This section provides insights and optimization tips for manufacturing efficiency.")

    # Optimization tips
    st.subheader("🎯 Optimization Tips")
    tips = [
        "**🌡️ Temperature Control**: Maintain injection temperature between 210-230°C for optimal material flow",
        "**⏱️ Cycle Time**: Target cycle times below 35 seconds for maximum throughput",
        "**🔧 Maintenance**: Schedule regular maintenance every 50-60 hours of operation",
        "**👨‍💼 Operator Training**: Experienced operators can improve output by 10-15%",
        "**🌡️ Ambient Conditions**: Keep ambient temperature stable around 24°C",
        "**⚙️ Machine Utilization**: Optimal range is 60-70% for balance between output and machine health",
        "**📊 Pressure Settings**: Higher pressure doesn't always mean better output - find the sweet spot",
        "**🔄 Material Quality**: Premium grade materials often yield 5-10% higher output"
    ]
    
    for tip in tips:
        st.markdown(f"• {tip}")

    # Sample data visualization (if data is available)
    try:
        # Try to load sample data for demonstration
        sample_df = pd.DataFrame({
            'Parameter': ['Temperature', 'Pressure', 'Cycle Time', 'Maintenance'],
            'Impact': [0.8, 0.6, 0.9, 0.7]
        })
        
        st.subheader("📊 Parameter Impact on Output")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Impact', y='Parameter', data=sample_df, ax=ax)
        ax.set_title('Relative Impact of Parameters on Output')
        st.pyplot(fig)
        
    except:
        # If visualization fails, continue without it
        pass

if __name__ == "__main__":
    main()

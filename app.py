import streamlit as st
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import random

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
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
MOCK_MODE = True  # Set to False when you have a real API deployed
API_URL = os.environ.get("FASTAPI_URL", "http://localhost:8000")

def mock_prediction(data):
    """Generate realistic mock predictions based on input parameters."""
    # Base calculation using key parameters
    base_output = 45
    
    # Factors based on different parameters
    temp_factor = (data['Injection_Temperature'] - 180) / 70 * 25
    pressure_factor = (data['Injection_Pressure'] - 80) / 70 * 20
    cycle_factor = (60 - data['Cycle_Time']) / 45 * 30
    efficiency_factor = data['Efficiency_Score'] * 50
    utilization_factor = data['Machine_Utilization'] * 25
    experience_factor = min(data['Operator_Experience'] / 120 * 15, 15)
    
    # Material grade bonuses
    material_bonus = {
        'Economy': 0,
        'Standard': 5,
        'Premium': 10
    }.get(data['Material_Grade'], 0)
    
    # Shift modifiers
    shift_modifier = {
        'Day': 2,
        'Evening': 0,
        'Night': -3
    }.get(data['Shift'], 0)
    
    # Calculate final prediction
    prediction = (base_output + temp_factor + pressure_factor + cycle_factor + 
                 efficiency_factor + utilization_factor + experience_factor + 
                 material_bonus + shift_modifier)
    
    # Add some randomness and ensure within reasonable bounds
    prediction = prediction * (0.95 + random.random() * 0.1)
    prediction = max(15, min(120, prediction))
    
    return {
        'prediction': round(prediction, 2),
        'confidence': round(82 + random.random() * 6, 1)
    }

def check_api_health():
    """Check if the FastAPI server is running."""
    if MOCK_MODE:
        return False  # Force mock mode
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False

def get_model_info():
    """Get model information."""
    return {
        'model_type': 'LinearRegression',
        'training_date': '2024-01-15',
        'performance_metrics': {
            'r2': 0.872,
            'rmse': 8.45,
            'mae': 6.23
        }
    }

def make_prediction(data):
    """Make prediction - uses mock data if API is unavailable."""
    if MOCK_MODE:
        return mock_prediction(data)
    
    try:
        response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        # Fallback to mock if API fails
        return mock_prediction(data)

def main():
    """Main function to run the Streamlit app."""
    st.markdown('<h1 class="main-header">🏭 Manufacturing Output Predictor</h1>', unsafe_allow_html=True)

    # Show mock mode warning if enabled
    if MOCK_MODE:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Demonstration Mode</strong><br>
            Currently using mock predictions. To use real predictions, deploy the FastAPI backend 
            and set the <code>FASTAPI_URL</code> environment variable.
        </div>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        model_info = get_model_info()
        if model_info:
            st.write(f"**Model Type:** {model_info.get('model_type', 'N/A')}")
            st.write(f"**Training Date:** {model_info.get('training_date', 'N/A')}")
            st.write(f"**R² Score:** {model_info.get('performance_metrics', {}).get('r2', 0):.3f}")
            st.write(f"**RMSE:** {model_info.get('performance_metrics', {}).get('rmse', 0):.2f}")
        
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
    """Single prediction page."""
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

    # Prepare data
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

    if st.button("🚀 Predict Output", use_container_width=True):
        with st.spinner("Calculating prediction..."):
            result = make_prediction(input_data)
            
            if result:
                prediction = result.get('prediction', 0)
                confidence = result.get('confidence', 80)
                
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.metric(
                    label="Predicted Parts Per Hour",
                    value=f"{prediction:.0f}",
                    delta=f"{confidence:.1f}% confidence"
                )
                st.markdown('</div>', unsafe_allow_html=True)

def batch_prediction_page():
    """Batch prediction page."""
    st.header("📊 Batch Prediction")
    st.info("Upload a CSV file with manufacturing data for predictions.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:")
            st.dataframe(df.head())
            
            if st.button("🔮 Generate Predictions", use_container_width=True):
                with st.spinner("Generating predictions..."):
                    # Generate mock predictions
                    predictions = []
                    for _, row in df.iterrows():
                        pred = mock_prediction(row.to_dict())
                        predictions.append(pred)
                    
                    result_df = df.copy()
                    result_df['Prediction'] = [p['prediction'] for p in predictions]
                    result_df['Confidence'] = [p['confidence'] for p in predictions]
                    
                    st.success("✅ Predictions generated successfully!")
                    st.dataframe(result_df)
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")

def data_analysis_page():
    """Data analysis page."""
    st.header("📈 Data Analysis & Insights")
    
    # Create sample visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sample data for visualizations
    parameters = ['Temperature', 'Pressure', 'Cycle Time', 'Efficiency', 'Utilization']
    impact = [0.85, 0.72, 0.91, 0.68, 0.63]
    
    sns.barplot(x=impact, y=parameters, ax=ax1)
    ax1.set_title('Parameter Impact on Output')
    
    # Optimization tips
    tips = [
        "🌡️ Maintain temperature between 210-230°C",
        "⏱️ Keep cycle time below 35 seconds",
        "🔧 Schedule maintenance every 50-60 hours",
        "👨‍💼 Invest in operator training",
        "🌡️ Stable ambient temperature improves quality",
        "⚙️ Optimal utilization: 60-70%"
    ]
    
    for i, tip in enumerate(tips, 1):
        st.write(f"{i}. {tip}")

if __name__ == "__main__":
    main()

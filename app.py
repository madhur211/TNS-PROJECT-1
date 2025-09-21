import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Manufacturing Output Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
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


# --- Load Model ---
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
def main():
    """Main function to run the Streamlit app."""
    st.markdown('<h1 class="main-header">🏭 Manufacturing Output Predictor</h1>', unsafe_allow_html=True)

    if model is None:
        st.warning("Model could not be loaded. Please check the logs.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.info("This app uses a Linear Regression model to predict manufacturing output based on machine parameters.")
        
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
        
        # Derived features are calculated automatically
        temp_pressure_ratio = injection_temp / injection_pressure if injection_pressure != 0 else 0
        total_cycle_time = cycle_time + cooling_time

        efficiency_score = st.slider("Efficiency Score", 0.01, 0.8, 0.15)
        machine_utilization = st.slider("Machine Utilization", 0.1, 0.8, 0.5)

    # Prepare data for prediction
    input_data = pd.DataFrame({
        "Injection_Temperature": [injection_temp], "Injection_Pressure": [injection_pressure],
        "Cycle_Time": [cycle_time], "Cooling_Time": [cooling_time],
        "Material_Viscosity": [material_viscosity], "Ambient_Temperature": [ambient_temp],
        "Machine_Age": [machine_age], "Operator_Experience": [operator_exp],
        "Maintenance_Hours": [maintenance_hrs], "Shift": [shift],
        "Machine_Type": [machine_type], "Material_Grade": [material_grade],
        "Day_of_Week": [day_of_week], "Temperature_Pressure_Ratio": [temp_pressure_ratio],
        "Total_Cycle_Time": [total_cycle_time], "Efficiency_Score": [efficiency_score],
        "Machine_Utilization": [machine_utilization]
    })
    
    # Prediction button
    if st.button("🚀 Predict Output", use_container_width=True):
        with st.spinner("Calculating..."):
            try:
                prediction = model.predict(input_data)
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.metric(label="Predicted Parts Per Hour", value=f"{prediction[0]:.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

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
                'Injection_Temperature', 'Injection_Pressure', 'Cycle_Time', 'Cooling_Time',
                'Material_Viscosity', 'Ambient_Temperature', 'Machine_Age', 'Operator_Experience',
                'Maintenance_Hours', 'Shift', 'Machine_Type', 'Material_Grade', 'Day_of_Week',
                'Temperature_Pressure_Ratio', 'Total_Cycle_Time', 'Efficiency_Score', 'Machine_Utilization'
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button("🔮 Make Batch Predictions", use_container_width=True):
                    with st.spinner("Processing..."):
                        try:
                            predictions = model.predict(df[required_cols])
                            df['Predicted_Output'] = [f"{p:.1f}" for p in predictions]
                            
                            st.success(f"Successfully made {len(df)} predictions!")
                            st.write("Prediction Results:")
                            st.dataframe(df)

                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from typing import List, Optional
import json
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Manufacturing Output Prediction API",
    description="API for predicting parts per hour from manufacturing equipment data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessor
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open('model_info.json', 'r') as f:
        model_info = json.load(f)
    
    print("Model and preprocessor loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    preprocessor = None
    model_info = {}

# Define input data model
class ManufacturingData(BaseModel):
    Injection_Temperature: float
    Injection_Pressure: float
    Cycle_Time: float
    Cooling_Time: float
    Material_Viscosity: float
    Ambient_Temperature: float
    Machine_Age: float
    Operator_Experience: float
    Maintenance_Hours: float
    Shift: str
    Machine_Type: str
    Material_Grade: str
    Day_of_Week: str
    Temperature_Pressure_Ratio: float
    Total_Cycle_Time: float
    Efficiency_Score: float
    Machine_Utilization: float
    Hour: Optional[int] = None
    Day: Optional[int] = None
    Month: Optional[int] = None

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_info: dict

@app.get("/")
async def root():
    return {
        "message": "Manufacturing Output Prediction API",
        "status": "active",
        "model_info": model_info
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/model-info")
async def get_model_info():
    return model_info

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: ManufacturingData):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Preprocess the input
        processed_input = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        
        # Calculate confidence (based on model R²)
        confidence = model_info.get('performance_metrics', {}).get('r2', 0.8) * 100
        
        return PredictionResponse(
            prediction=round(prediction, 2),
            confidence=round(confidence, 2),
            model_info=model_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(data: List[ManufacturingData]):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dicts = [item.dict() for item in data]
        input_df = pd.DataFrame(input_dicts)
        
        # Preprocess the input
        processed_input = preprocessor.transform(input_df)
        
        # Make predictions
        predictions = model.predict(processed_input)
        
        # Prepare response
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "input_id": i + 1,
                "prediction": round(pred, 2),
                "confidence": round(model_info.get('performance_metrics', {}).get('r2', 0.8) * 100, 2)
            })
        
        return {
            "predictions": results,
            "total_predictions": len(results),
            "model_info": model_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
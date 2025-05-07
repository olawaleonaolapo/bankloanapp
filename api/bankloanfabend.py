from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

# Initialize FastAPI app
app = FastAPI(title="Bank Loan Approval Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production, e.g., ["https://your-app.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set working directory to project root
PATH_TO_THIS_FILE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(PATH_TO_THIS_FILE))
print("CURRENT WORKING DIRECTORY:", os.getcwd())

# Define feature lists
numerical_features = ["Income", "Experience", "CCAvg", "Family"]
categorical_features = [
    "Securities.Account", "Certificate.Deposit.Account", "Online", "CreditCard",
    "Mortgage.Category", "Education", "ZIP_90", "ZIP_91", "ZIP_92", "ZIP_93", "ZIP_94", "ZIP_95"
]
all_features = numerical_features + categorical_features

# Load preprocessor and models
try:
    preprocessor = joblib.load(os.path.join(PATH_TO_THIS_FILE, "../preprocessor.joblib"))
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Preprocessor file 'preprocessor.joblib' not found.")

model_files = {
    "Logistic Regression": "Logistic Regression_best_model.joblib",
    "Decision Tree": "Decision Tree_best_model.joblib",
    "Random Forest": "Random Forest_best_model.joblib",
    "Gradient Boosting": "Gradient Boosting_best_model.joblib"
}
models = {}
for name, file in model_files.items():
    try:
        models[name] = joblib.load(os.path.join(PATH_TO_THIS_FILE, f"../{file}"))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Model file '{file}' not found.")

# Check HTML file
html_file = os.path.join(PATH_TO_THIS_FILE, "../public/bankloanindex.html")
if not os.path.isfile(html_file):
    raise HTTPException(status_code=500, detail="HTML file 'bankloanindex.html' not found.")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Server is running. Access the application at this domain."}

# Serve front-end
@app.get("/")
async def serve_frontend():
    print(f"Serving HTML file: {html_file}")
    return FileResponse(html_file)

# Prediction endpoint
@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        if data.model not in models:
            raise HTTPException(status_code=400, detail=f"Invalid model choice: {data.model}. Available models: {list(models.keys())}")
        print("Received inputData:", data.inputData)
        input_df = pd.DataFrame([data.inputData], columns=all_features)
        missing_cols = set(all_features) - set(data.inputData.keys())
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing input features: {missing_cols}")
        input_scaled = preprocessor.transform(input_df)
        model = models[data.model]
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        print(f"Prediction for {data.model}: {prediction}, Probabilities: {probabilities}")
        return {
            "prediction": int(prediction),
            "probabilities": probabilities.tolist(),
            "model": data.model
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Pydantic model
class PredictionInput(BaseModel):
    inputData: dict
    model: str
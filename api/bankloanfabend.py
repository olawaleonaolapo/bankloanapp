from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Bank Loan Approval Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PATH_TO_THIS_FILE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(PATH_TO_THIS_FILE))
print("CURRENT WORKING DIRECTORY:", os.getcwd())

numerical_features = ["Income", "Experience", "CCAvg", "Family"]
categorical_features = [
    "Securities.Account",
    "Certificate.Deposit.Account",
    "Online",
    "CreditCard",
    "Mortgage.Category",
    "Education",
    "ZIP_90",
    "ZIP_91",
    "ZIP_92",
    "ZIP_93",
    "ZIP_94",
    "ZIP_95",
]
all_features = numerical_features + categorical_features

try:
    preprocessor = joblib.load(os.path.join(PATH_TO_THIS_FILE, "../preprocessor.joblib"))
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Preprocessor file not found.")

model_files = {
    "Logistic Regression": "Logistic Regression_best_model.joblib",
    "Decision Tree": "Decision Tree_best_model.joblib",
    "Random Forest": "Random Forest_best_model.joblib",
    "Gradient Boosting": "Gradient Boosting_best_model.joblib",
}
models = {}
for name, file in model_files.items():
    try:
        models[name] = joblib.load(os.path.join(PATH_TO_THIS_FILE, f"../{file}"))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Model file '{file}' not found.")

html_file = os.path.join(PATH_TO_THIS_FILE, "../public/bankloanindex.html")
if not os.path.isfile(html_file):
    raise HTTPException(status_code=500, detail="HTML file not found.")

class PredictionInput(BaseModel):
    inputData: dict
    model: str

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Server is running. Access the application at this domain.",
        "models_loaded": list(models.keys()),
    }

@app.get("/")
async def serve_frontend():
    print(f"Serving HTML file: {html_file}")
    return FileResponse(html_file)

@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        if data.model not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model choice: {data.model}. Available models: {list(models.keys())}",
            )
        print("Received inputData:", data.inputData)
        print("Input data types:", {k: type(v).__name__ for k, v in data.inputData.items()})
        input_df = pd.DataFrame([data.inputData], columns=all_features)
        missing_cols = set(all_features) - set(data.inputData.keys())
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing input features: {missing_cols}")
        print("Input DataFrame:", input_df)
        input_scaled = preprocessor.transform(input_df)
        model = models[data.model]
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        print(f"Prediction for {data.model}: {prediction}, Probabilities: {probabilities}")
        return {
            "loan_status": "Approved" if prediction == 1 else "Rejected",
            "probability_approved": float(probabilities[1]),
            "probability_rejected": float(probabilities[0]),
            "model_used": data.model,
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
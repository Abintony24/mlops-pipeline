import mlflow.sklearn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Iris MLOps Level 3 API")

MODEL_PATH = "model_export_v2"

model = None


# Load exported MLflow model
def load_model():

    try:
        loaded_model = mlflow.sklearn.load_model(MODEL_PATH)

        print("Model loaded successfully.")

        return loaded_model

    except Exception as e:
        print(f"Model loading failed: {e}")
        return None


# Load model when application starts
model = load_model()


class IrisInput(BaseModel):

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Health check endpoint
@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable."
        )

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }
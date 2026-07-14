# 1. Let's get the file path of the root first and then get to the model
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH =  PROJECT_ROOT / "models" / "tfidf_linear_svm.joblib"


# 2. We need FastAPI application
app = FastAPI(
    title="Financial Intelligence System API",
    version="1.0.0"
)

# 3. Let's load the model
model = joblib.load(MODEL_PATH)

# These are the commonly used variables
MODEL_NAME = "TF-IDF + Linear SVM"
MODEL_VERSION = "v1-baseline"

# --------- API Endpoints ---------------
# To get the status of the API
@app.get("/health")
def health_check():
    return{
        "status": "ok",
        "model_loaded": model is not None
    }

# To get the model info
@app.get("/model_info")
def model_info():
    return{
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_path": str(MODEL_PATH),
        "classes": model.classes_.tolist()
    }

# To predict a text, we need the text from the user -> so we need to validate the data before predicting
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Financial news headline"

    )

class PredictResponse(BaseModel):
    sentiment:str
    model_name:str
    model_version:str


# Post request for predict
@app.post("/predict", response_model=PredictResponse)
def predict(request:PredictRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Input text cannot contain empty string"
        )
    prediction = model.predict([text])[0]
    return{
        "sentiment":prediction,
        "model_name":MODEL_NAME,
        "model_version":MODEL_VERSION
    }


# Now, we need to do the same for Batch of texts

class BatchPredictRequest(BaseModel):
    texts:list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of Financial news headline"
    )

class BatchPredictResponse(BaseModel):
    predictions: list[str]
    model_name: str
    model_version: str

# Post request for BatchPrediction
@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(request:BatchPredictRequest):
    cleaned_texts = [text.strip() for text in request.texts]
    # Check if there are any empty texts in the list, if it even has one raise an error
    if any (not text for text in cleaned_texts):
        raise HTTPException(
            status_code=400,
            detail="Input texts cannot be empty strings"
        )
    
    # response woould be in numpy arrays which json can't deserialize and we convert them to a normal list
    predictions = model.predict(cleaned_texts).tolist()

    return{
        "predictions": predictions,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION
    }





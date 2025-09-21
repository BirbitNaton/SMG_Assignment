import logging

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.models import get_model_info
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PyFuncModel
from pydantic import ValidationError

from task1.api.schemas import BatchPredictionInput, PredictionInput
from task1.api.utils import MODEL_NAME, MODEL_STAGE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("catboost_api")

mlflow_client = mlflow.tracking.MlflowClient()

# Load model at startup
app = FastAPI(title="CatBoost House Price Predictor API")


def load_model_for_app() -> tuple[PyFuncModel | None, ModelInfo | None]:
    """Load the MLflow model and its information for the FastAPI app.

    Retrieves the latest model version from MLflow and loads it
    into the application and loads its metadata.

    Returns
    -------
    tuple[PyFuncModel | None, ModelInfo | None]
        The loaded model and its metadata, or None if loading failed.

    Raises
    ------
    ValueError
        When no model versions are found or loading fails.
    """
    try:
        versions = mlflow_client.get_latest_versions(MODEL_NAME)
        if not versions:
            raise ValueError(
                f"No model versions found for {MODEL_NAME} at stage {MODEL_STAGE}"
            )

        model_version = versions[0].version
        model_uri = f"models:/{MODEL_NAME}/{model_version}"

        model = mlflow.pyfunc.load_model(model_uri)
        model_info = get_model_info(model_uri)
        logger.info(f"Loaded model {MODEL_NAME} version {model_version}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        model_info = None

    return model, model_info


model, model_info = load_model_for_app()


@app.get("/health")
def health():
    if model is None:
        model_loc, _ = load_model_for_app()
        if model_loc is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/model/info")
def model_info_endpoint():
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model info not available")
    return {
        "model_name": model_info.name,
        "params": model_info.params,
        "latest_version": mlflow_client.get_latest_versions(MODEL_NAME)[0].version,
        "creation_timestamp": model_info.creation_timestamp,
    }


@app.post("/predict")
def predict(input: PredictionInput):
    try:
        df = pd.DataFrame([input.model_dump()])
        prediction = np.exp(model.predict(df))
        return {"prediction": prediction[0]}
    except ValidationError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/batch_predict")
def batch_predict(batch: BatchPredictionInput):
    try:
        df = pd.DataFrame([item.model_dump() for item in batch.inputs])
        predictions = np.exp(model.predict(df))
        return {"predictions": predictions.tolist()}
    except ValidationError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")

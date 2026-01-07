import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Breast Cancer Prediction API")

MODEL_PATH = os.path.join("models", "model.joblib")
model = joblib.load(MODEL_PATH)


class InputData(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Breast Cancer API is running. Use /docs"}


@app.get("/model-info")
def model_info():
    classes = getattr(model, "classes_", None)
    if classes is not None:
        # numpy array -> JSON serializable list
        classes = classes.tolist()

    return {
        "model_path": MODEL_PATH,
        "model_type": str(type(model)),
        "n_features_in_": getattr(model, "n_features_in_", None),
        "classes_": classes,
        "has_predict_proba": hasattr(model, "predict_proba"),
    }


@app.post("/predict")
def predict(data: InputData):
    # 1) Basic validation
    if data.features is None:
        raise HTTPException(status_code=400, detail="Missing 'features' field")

    # 2) Validate number of features (must be 30 for your model)
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and len(data.features) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected} features, got {len(data.features)}",
        )

    # 3) Predict
    X = np.array(data.features, dtype=float).reshape(1, -1)
    pred = int(model.predict(X)[0])

    # 4) Probability for predicted class (robust)
    proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]  # aligned with model.classes_
        classes = getattr(model, "classes_", None)

        if classes is not None:
            idx = int(np.where(classes == pred)[0][0])
            proba = float(probs[idx])
        else:
            proba = float(np.max(probs))

    return {"prediction": pred, "probability": proba}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)

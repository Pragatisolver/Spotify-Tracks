"""FastAPI app for genre prediction and song recommendations."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils import load_joblib  # noqa: E402
from src.recommendation_engine import SongRecommender  # noqa: E402

app = FastAPI(title="Spotify Genre Classification & Recommendation API")

def resolve_artifact_path(filename: str) -> Path:
    """Support both saved_models and legacy saved_model directories."""
    candidate_dirs = [
        ROOT_DIR / "models" / "deploy",
        ROOT_DIR / "models" / "saved_models",
        ROOT_DIR / "models" / "saved_model",
    ]
    for candidate in candidate_dirs:
        path = candidate / filename
        if path.exists():
            return path
    return candidate_dirs[0] / filename


MODEL_PATH = resolve_artifact_path("best_genre_model.joblib")
RECOMMENDER_PATH = resolve_artifact_path("song_recommender.joblib")

model_bundle: dict | None = None
recommender: SongRecommender | None = None


class GenreRequest(BaseModel):
    danceability: float = Field(..., ge=0, le=1)
    energy: float = Field(..., ge=0, le=1)
    loudness: float
    speechiness: float = Field(..., ge=0, le=1)
    acousticness: float = Field(..., ge=0, le=1)
    instrumentalness: float = Field(..., ge=0, le=1)
    liveness: float = Field(..., ge=0, le=1)
    valence: float = Field(..., ge=0, le=1)
    tempo: float = Field(..., gt=0)
    popularity: float | None = 0
    duration_ms: float | None = 0
    explicit: float | None = 0
    key: float | None = 0
    mode: float | None = 0
    time_signature: float | None = 4


class RecommendBySongRequest(BaseModel):
    song_name: str
    top_k: int = Field(default=5, ge=1, le=20)


class RecommendByFeaturesRequest(BaseModel):
    features: dict[str, float]
    top_k: int = Field(default=5, ge=1, le=20)


@app.on_event("startup")
def load_artifacts() -> None:
    """Load trained artifacts once when API starts."""
    global model_bundle, recommender

    if MODEL_PATH.exists():
        model_bundle = load_joblib(MODEL_PATH)

    if RECOMMENDER_PATH.exists():
        payload = load_joblib(RECOMMENDER_PATH)
        recommender = SongRecommender.from_artifacts(payload)


@app.get("/health")
def health() -> dict[str, str]:
    """Basic health endpoint."""
    return {"status": "ok"}


@app.post("/predict-genre")
def predict_genre(request: GenreRequest) -> dict[str, str]:
    """Predict song genre from input audio features."""
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model artifact not found. Train model first.")

    feature_columns: list[str] = model_bundle["feature_columns"]
    selected_names: list[str] = model_bundle["selected_feature_names"]

    values = request.model_dump()
    row = {col: float(values.get(col, 0.0)) for col in feature_columns}
    df = pd.DataFrame([row])

    if selected_names:
        df = df[selected_names]

    scaler = model_bundle["scaler"]
    X_scaled = scaler.transform(df)

    pca = model_bundle.get("pca")
    if pca is not None:
        X_scaled = pca.transform(X_scaled)

    model = model_bundle["model"]
    pred_encoded = model.predict(X_scaled)

    label_encoder = model_bundle["label_encoder"]
    pred_label = label_encoder.inverse_transform(pred_encoded)[0]

    return {"predicted_genre": str(pred_label), "model": str(model_bundle["model_name"])}


@app.post("/recommend")
def recommend_song(request: RecommendBySongRequest) -> dict[str, object]:
    """Recommend top-k similar songs by song name."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender artifact not found. Train model first.")

    try:
        recs = recommender.recommend_by_song(request.song_name, top_k=request.top_k)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err

    return {"input_song": request.song_name, "recommendations": recs}


@app.post("/recommend-by-features")
def recommend_from_features(request: RecommendByFeaturesRequest) -> dict[str, object]:
    """Recommend top-k songs using input audio features."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender artifact not found. Train model first.")

    recs = recommender.recommend_by_features(request.features, top_k=request.top_k)
    return {"recommendations": recs}

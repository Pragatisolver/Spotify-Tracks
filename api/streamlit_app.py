"""Streamlit UI for Spotify genre prediction and song recommendation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.recommendation_engine import SongRecommender
from src.utils import load_joblib


def resolve_artifact_path(filename: str) -> Path:
    """Support both saved_models and legacy saved_model directories."""
    candidate_dirs = [
        ROOT / "models" / "deploy",
        ROOT / "models" / "saved_models",
        ROOT / "models" / "saved_model",
    ]
    for candidate in candidate_dirs:
        path = candidate / filename
        if path.exists():
            return path
    return candidate_dirs[0] / filename


MODEL_PATH = resolve_artifact_path("best_genre_model.joblib")
RECOMMENDER_PATH = resolve_artifact_path("song_recommender.joblib")
METRICS_PATH = resolve_artifact_path("all_model_metrics.json")
PLOT_MODEL_COMPARE = resolve_artifact_path("model_comparison.png")
PLOT_CONFUSION = resolve_artifact_path("best_model_confusion_matrix.png")

st.set_page_config(
    page_title="SongSage",
    page_icon="🎧",
    layout="wide",
)

st.markdown(
    """
<style>
.main {
    background: radial-gradient(circle at 10% 20%, #f4f7ff 0%, #f8f9fc 40%, #ffffff 100%);
}
.block-container {
    padding-top: 2rem;
}
.hero {
    padding: 1.4rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 100%);
    color: white;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}
.hero h1 {
    margin: 0;
    font-size: 2rem;
    letter-spacing: 0.3px;
}
.hero p {
    margin: 0.6rem 0 0;
    opacity: 0.95;
    font-size: 1rem;
}
.metric-card {
    border: 1px solid rgba(12, 74, 110, 0.12);
    border-radius: 14px;
    padding: 0.8rem 1rem;
    background: #ffffff;
}
.metric-label {
    color: #475569;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.metric-value {
    color: #0f172a;
    font-size: 1.05rem;
    font-weight: 700;
}
.section-title {
    margin-top: 0.2rem;
    margin-bottom: 0.4rem;
}
.small-note {
    color: #334155;
    font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_genre_bundle() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    return load_joblib(MODEL_PATH)


@st.cache_resource
def load_recommender() -> SongRecommender | None:
    if not RECOMMENDER_PATH.exists():
        return None
    payload = load_joblib(RECOMMENDER_PATH)
    return SongRecommender.from_artifacts(payload)


@st.cache_data
def load_metrics() -> dict[str, Any] | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def build_song_option_map(catalog: pd.DataFrame) -> dict[str, str]:
    if catalog.empty or "track_name" not in catalog.columns:
        return {}

    options: dict[str, str] = {}
    for _, row in catalog.iterrows():
        song = str(row.get("track_name", "Unknown")).strip()
        artist = str(row.get("artists", "Unknown")).strip()
        genre = str(row.get("track_genre", "Unknown")).strip()
        key = f"{song} | {artist} | {genre}"
        options[key] = song
    return options


AUDIO_PRESETS: dict[str, dict[str, float]] = {
    "Balanced Pop": {
        "danceability": 0.70,
        "energy": 0.70,
        "loudness": -6.0,
        "speechiness": 0.06,
        "acousticness": 0.18,
        "instrumentalness": 0.0,
        "liveness": 0.15,
        "valence": 0.65,
        "tempo": 120.0,
    },
    "Acoustic Chill": {
        "danceability": 0.45,
        "energy": 0.35,
        "loudness": -13.0,
        "speechiness": 0.04,
        "acousticness": 0.82,
        "instrumentalness": 0.1,
        "liveness": 0.12,
        "valence": 0.42,
        "tempo": 94.0,
    },
    "High Energy EDM": {
        "danceability": 0.78,
        "energy": 0.92,
        "loudness": -4.0,
        "speechiness": 0.07,
        "acousticness": 0.05,
        "instrumentalness": 0.45,
        "liveness": 0.22,
        "valence": 0.60,
        "tempo": 128.0,
    },
}


def predict_genre(bundle: dict[str, Any], values: dict[str, float]) -> str:
    feature_columns = bundle["feature_columns"]
    selected_feature_names = bundle["selected_feature_names"]

    row = {col: float(values.get(col, 0.0)) for col in feature_columns}
    X = pd.DataFrame([row])

    if selected_feature_names:
        X = X[selected_feature_names]

    scaler = bundle["scaler"]
    X_scaled = scaler.transform(X)

    pca = bundle.get("pca")
    if pca is not None:
        X_scaled = pca.transform(X_scaled)

    model = bundle["model"]
    pred = model.predict(X_scaled)
    label_encoder = bundle["label_encoder"]
    return str(label_encoder.inverse_transform(pred)[0])


def humanize_model_name(name: str) -> str:
    """Convert snake_case model names to readable labels."""
    cleaned = name.replace("_deploy", "").replace("_", " ").strip()
    return cleaned.title()


st.markdown(
    """
<div class="hero">
  <h1>SongSage</h1>
  <p>Predict song genres and discover similar tracks using audio features.</p>
</div>
""",
    unsafe_allow_html=True,
)

bundle = load_genre_bundle()
recommender = load_recommender()
metrics = load_metrics()

if bundle is None or recommender is None:
    st.error("Model artifacts are missing. Run `python main.py train` first.")
    st.stop()

best_name = max(metrics, key=lambda m: metrics[m]["f1"]) if metrics else "unknown"
best_f1 = f"{metrics[best_name]['f1']:.3f}" if metrics else "N/A"
best_accuracy = f"{metrics[best_name]['accuracy']:.3f}" if metrics else "N/A"
best_precision = f"{metrics[best_name]['precision']:.3f}" if metrics else "N/A"
best_recall = f"{metrics[best_name]['recall']:.3f}" if metrics else "N/A"
best_model_label = humanize_model_name(best_name)

tab1, tab2, tab3 = st.tabs(["Genre Prediction", "Song Recommender", "Model Insights"])

with tab1:
    st.markdown(
        '<h3 class="section-title">Genre Prediction</h3>', unsafe_allow_html=True
    )
    st.caption("Use the preset for a quick demo, or adjust each control manually.")

    preset_col, _ = st.columns([1, 2])
    with preset_col:
        preset_name = st.selectbox("Quick Preset", list(AUDIO_PRESETS.keys()), index=0)
    preset = AUDIO_PRESETS[preset_name]

    left, middle, right = st.columns(3)
    with left:
        danceability = st.slider(
            "Danceability", 0.0, 1.0, float(preset["danceability"])
        )
        energy = st.slider("Energy", 0.0, 1.0, float(preset["energy"]))
        loudness = st.slider("Loudness", -60.0, 5.0, float(preset["loudness"]))
        speechiness = st.slider("Speechiness", 0.0, 1.0, float(preset["speechiness"]))
        acousticness = st.slider(
            "Acousticness", 0.0, 1.0, float(preset["acousticness"])
        )
    with middle:
        instrumentalness = st.slider(
            "Instrumentalness", 0.0, 1.0, float(preset["instrumentalness"])
        )
        liveness = st.slider("Liveness", 0.0, 1.0, float(preset["liveness"]))
        valence = st.slider("Valence", 0.0, 1.0, float(preset["valence"]))
        tempo = st.slider("Tempo", 40.0, 240.0, float(preset["tempo"]))
    with right:
        popularity = st.slider("Popularity", 0, 100, 50)
        duration_ms = st.slider("Duration (ms)", 30000, 600000, 200000, step=1000)
        explicit = st.selectbox("Explicit", [0, 1], index=0)
        key = st.slider("Key", 0, 11, 5)
        mode = st.selectbox("Mode", [0, 1], index=1)
        time_signature = st.selectbox("Time Signature", [3, 4, 5, 6, 7], index=1)

    if st.button("Predict Genre", type="primary", use_container_width=True):
        features = {
            "danceability": danceability,
            "energy": energy,
            "loudness": loudness,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo,
            "popularity": popularity,
            "duration_ms": duration_ms,
            "explicit": explicit,
            "key": key,
            "mode": mode,
            "time_signature": time_signature,
        }
        genre = predict_genre(bundle, features)
        st.success(f"Predicted Genre: **{genre}**")

with tab2:
    st.markdown(
        '<h3 class="section-title">Song Recommender</h3>', unsafe_allow_html=True
    )
    st.caption("Search by title/artist, then generate top similar songs.")

    catalog = recommender.catalog if recommender.catalog is not None else pd.DataFrame()
    option_map = build_song_option_map(catalog)
    labels = sorted(option_map.keys())

    if not labels:
        st.warning("No songs available in recommender catalog.")
    else:
        selected_label = st.selectbox("Song (Title | Artist | Genre)", labels, index=0)
        top_k = st.slider("Recommendations", 1, 10, 5)

        if st.button("Get Recommendations", type="primary", use_container_width=True):
            song_name = option_map[selected_label]
            try:
                recs = recommender.recommend_by_song(song_name, top_k=top_k)
                st.success(f"Top {len(recs)} recommendations for: **{song_name}**")
                result_df = pd.DataFrame(recs)
                if not result_df.empty and "similarity_score" in result_df.columns:
                    result_df["similarity_score"] = result_df["similarity_score"].round(
                        4
                    )
                st.dataframe(result_df, use_container_width=True, hide_index=True)
            except ValueError as err:
                st.error(str(err))

with tab3:
    st.markdown('<h3 class="section-title">Model Insights</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="small-note">Inspect evaluation charts and final model performance.</p>',
        unsafe_allow_html=True,
    )
    row1c1, row1c2, row1c3 = st.columns(3)
    with row1c1:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Model</div>
  <div class="metric-value">{best_model_label}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with row1c2:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Accuracy</div>
  <div class="metric-value">{(float(best_accuracy) * 100):.1f}%</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with row1c3:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">F1 Score</div>
  <div class="metric-value">{best_f1}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    row2c1, row2c2, row2c3 = st.columns(3)
    with row2c1:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Precision</div>
  <div class="metric-value">{best_precision}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with row2c2:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Recall</div>
  <div class="metric-value">{best_recall}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with row2c3:
        st.markdown(
            """
<div class="metric-card">
  <div class="metric-label">Dataset</div>
  <div class="metric-value">Spotify Tracks Dataset</div>
</div>
""",
            unsafe_allow_html=True,
        )

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        if PLOT_MODEL_COMPARE.exists():
            st.image(
                str(PLOT_MODEL_COMPARE),
                caption="Model Comparison",
                use_container_width=True,
            )
        else:
            st.info("Model comparison plot not found.")
    with plot_col2:
        if PLOT_CONFUSION.exists():
            st.image(
                str(PLOT_CONFUSION),
                caption="Best Model Confusion Matrix",
                use_container_width=True,
            )
        else:
            st.info("Confusion matrix plot not found.")

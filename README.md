# SongSage 🎧

Predict song genres and discover similar tracks using audio features.

SongSage is an end-to-end machine learning application built on Spotify audio data. It combines genre classification and song recommendation in a production-style project with modular ML code, artifact persistence, API endpoints, and an interactive Streamlit UI.

## Live Demo

[https://gax4gvgkau3ey3dyza5uiz.streamlit.app](https://gax4gvgkau3ey3dyza5uiz.streamlit.app)

## Core Features

- **Genre Prediction** from song-level audio features
- **Song Recommendation** using nearest neighbors + cosine similarity
- **Model Insights** with core performance metrics and evaluation charts

## Dataset

- **Source:** Spotify Tracks Dataset (Kaggle)
- **Target:** `track_genre`
- **Core audio features:** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
- **Additional features (if available):** `popularity`, `duration_ms`, `explicit`, `key`, `mode`, `time_signature`

## ML Approach

### Genre Classification
Models trained and compared:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost (optional)

Metrics used:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Recommendation Engine
- Standardized feature vectors
- Nearest Neighbors algorithm
- Cosine similarity ranking

## Project Structure

```text
spotify_ml_project/
├── data/
├── notebooks/
├── src/
├── models/
│   ├── deploy/
│   ├── saved_model/
│   └── saved_models/
├── api/
│   ├── app.py
│   └── streamlit_app.py
├── requirements.txt
├── main.py
└── README.md
```

## Local Setup

```bash
cd spotify_ml_project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place dataset CSV in:

```text
data/raw/
```

## Train

```bash
python main.py train
```

Optional:

```bash
python main.py train --k-best 10 --use-pca --tune
python main.py train --data-csv /absolute/path/to/dataset.csv
```

## Run App

```bash
streamlit run api/streamlit_app.py
```

## Run API

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Key Endpoints

- `POST /predict-genre`
- `POST /recommend`
- `POST /recommend-by-features`

## Deployment Artifacts

Deployment-ready files are stored in:

```text
models/deploy/
```

Includes:
- `best_genre_model.joblib`
- `song_recommender.joblib`
- `all_model_metrics.json`
- `model_comparison.png`
- `best_model_confusion_matrix.png`

## Author

**Pragati Godara**

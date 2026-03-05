# SongSage

**Discover genres and find your next favorite track.**

SongSage is an end-to-end machine learning project built on Spotify audio features.  
It combines two practical capabilities in one product:

- Genre prediction from song-level audio signals
- Similar-song recommendation using nearest-neighbor similarity

This project is designed like a production-ready ML application, with modular training code, artifact persistence, a FastAPI backend, and a polished Streamlit interface.

---

## Why I Built This

I wanted to move beyond a notebook-only ML prototype and build something people can actually use.

SongSage reflects that goal:
- clean project structure
- reproducible training pipeline
- explainable evaluation outputs
- an interface that feels like a real product, not a demo script

---

## Dataset

- **Source:** [Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset)
- **Target Column:** `track_genre`
- **Core audio features used:**
  - `danceability`
  - `energy`
  - `loudness`
  - `speechiness`
  - `acousticness`
  - `instrumentalness`
  - `liveness`
  - `valence`
  - `tempo`

Additional metadata/features are used where available (`popularity`, `duration_ms`, `explicit`, `key`, `mode`, `time_signature`).

---

## What SongSage Does

### 1. Genre Classification
Trains and compares multiple classifiers:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost (optional if available)

Evaluation includes:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 2. Song Recommendation
Builds a recommendation engine using:
- Standardized feature vectors
- Nearest Neighbors with cosine similarity

User can:
- input a song name
- receive top-k similar songs

---

## Project Structure

```text
spotify_ml_project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── recommendation_engine.py
│   └── utils.py
├── models/
│   ├── deploy/                  # lightweight deployment artifacts
│   ├── saved_model/             # legacy compatibility
│   └── saved_models/            # full local training artifacts
├── api/
│   ├── app.py                   # FastAPI backend
│   └── streamlit_app.py         # SongSage frontend
├── requirements.txt
├── README.md
└── main.py
```

---

## Local Setup

```bash
cd /Users/pragatigodara/Documents/Playground/spotify_ml_project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place your dataset CSV in:

```text
data/raw/
```

---

## Train the System

```bash
python main.py train
```

Optional flags:

```bash
python main.py train --k-best 10 --use-pca --tune
python main.py train --data-csv /absolute/path/to/dataset.csv
```

This step:
- preprocesses data
- trains classifier models
- picks best model
- saves evaluation artifacts
- builds recommender artifacts
- creates deploy-friendly artifacts in `models/deploy/`

---

## Run SongSage UI

```bash
streamlit run api/streamlit_app.py
```

What users see:
- **Genre Prediction** tab with guided feature controls
- **Song Recommender** tab with top-k recommendations
- **Model Insights** tab with charts and compact performance cards

---

## Run API

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

### Key Endpoints

#### `POST /predict-genre`
Input: audio feature JSON  
Output: predicted genre label

#### `POST /recommend`
Input:
```json
{
  "song_name": "Blinding Lights",
  "top_k": 5
}
```
Output: top similar songs

#### `POST /recommend-by-features`
Input: custom feature vector  
Output: top similar songs

---

## Model Artifacts

### Deployment-ready artifacts
Stored in:

```text
models/deploy/
```

Includes:
- `best_genre_model.joblib`
- `song_recommender.joblib`
- `all_model_metrics.json`
- `model_comparison.png`
- `best_model_confusion_matrix.png`

---

## Deployment Notes (Streamlit Cloud)

For successful deployment:
- ensure `models/deploy/*` files are committed and pushed
- set app entry point to:
  - `api/streamlit_app.py`
- reboot app after each push

If you see "Model artifacts are missing", check whether deploy artifacts exist in GitHub repo, not just locally.

---

## Engineering Highlights

- Modular Python code with type hints and docstrings
- Reusable preprocessing + feature engineering pipeline
- Automated model comparison and artifact persistence
- API + UI separation for clean product architecture
- Backward-compatible artifact path resolution (`deploy`, `saved_models`, `saved_model`)

---

## EDA Notebook

Open:

```text
notebooks/exploratory_data_analysis.ipynb
```

Includes:
- genre distribution
- audio feature distributions
- correlation heatmap
- feature importance analysis

---

## Current Limitations

- Genre prediction is a high-class, imbalanced task; baseline performance is expectedly moderate.
- Confusion matrix can be visually dense due to many genres.
- Recommendation quality depends on feature similarity, not user listening history.

---

## Next Improvements

- stronger model tuning and class balancing
- top-k genre confidence outputs
- cleaner reduced-label confusion visualization
- richer recommendation ranking (hybrid similarity + popularity signals)
- CI checks and containerized deployment

---

## Author Note

Built with intent to feel both **engineering-grade** and **user-friendly**.

If you use this project, I recommend treating it as a strong baseline for a real music intelligence product and iterating on model quality + UX together.

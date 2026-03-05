# Spotify ML Studio

An end-to-end machine learning project built on the Spotify Tracks dataset.

This project gives users two capabilities:
1. Predict song genre from audio features
2. Recommend similar songs from a selected track

Dataset: [Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset)

## What Users Get

- Multi-model genre classifier:
  - Logistic Regression
  - Random Forest
  - SVM
  - XGBoost (if installed)
- Automatic model comparison and best-model selection
- Recommendation engine using cosine similarity + nearest neighbors
- FastAPI backend for API-based usage
- Streamlit frontend for easy interactive usage
- EDA notebook with core visualizations

## Project Layout

```text
spotify_ml_project/
├── data/
│   ├── raw/                     # Put Kaggle CSV here
│   └── processed/
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── recommendation_engine.py
│   └── utils.py
├── models/
│   └── saved_models/            # Trained artifacts + plots
├── api/
│   ├── app.py                   # FastAPI app
│   └── streamlit_app.py         # User-friendly frontend
├── requirements.txt
├── README.md
└── main.py
```

## Quick Start (Recommended)

### 1. Setup

```bash
cd /Users/pragatigodara/Documents/Playground/spotify_ml_project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add Dataset

Download Kaggle CSV and place it in:

`/Users/pragatigodara/Documents/Playground/spotify_ml_project/data/raw`

### 3. Train Models

```bash
python main.py train
```

Optional (bonus):

```bash
python main.py train --k-best 10 --use-pca --tune
```

### 4. Run User Interface (Best for non-technical users)

```bash
streamlit run api/streamlit_app.py
```

Open the URL shown in terminal (usually `http://localhost:8501`).

## Streamlit Screens

### Predict Genre
- Input audio features using sliders
- Click `Predict Genre`
- See predicted class instantly

### Recommend Songs
- Select an existing song
- Choose top-k recommendations
- View similar songs in a table

### Model Insights
- See metrics for each algorithm
- View model comparison plot
- View confusion matrix for best model

## API Usage

Run API:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI:

`http://127.0.0.1:8000/docs`

### Endpoint: `POST /predict-genre`

Example payload:

```json
{
  "danceability": 0.72,
  "energy": 0.81,
  "loudness": -5.1,
  "speechiness": 0.04,
  "acousticness": 0.10,
  "instrumentalness": 0.00,
  "liveness": 0.12,
  "valence": 0.74,
  "tempo": 122.5,
  "popularity": 80,
  "duration_ms": 203000,
  "explicit": 0,
  "key": 5,
  "mode": 1,
  "time_signature": 4
}
```

### Endpoint: `POST /recommend`

```json
{
  "song_name": "Blinding Lights",
  "top_k": 5
}
```

### Endpoint: `POST /recommend-by-features`

```json
{
  "features": {
    "danceability": 0.72,
    "energy": 0.81,
    "loudness": -5.1,
    "speechiness": 0.04,
    "acousticness": 0.10,
    "instrumentalness": 0.00,
    "liveness": 0.12,
    "valence": 0.74,
    "tempo": 122.5
  },
  "top_k": 5
}
```

## Output Artifacts

After training, these are generated in `models/saved_models/`:

- `best_genre_model.joblib`
- `song_recommender.joblib`
- `all_model_metrics.json`
- `model_comparison.png`
- `best_model_confusion_matrix.png`

## EDA Notebook

Use:

`/Users/pragatigodara/Documents/Playground/spotify_ml_project/notebooks/exploratory_data_analysis.ipynb`

Includes:
- Genre distribution
- Audio feature distributions
- Correlation heatmap
- Feature importance chart

## Troubleshooting

- `Model artifact not found`:
  - Run `python main.py train`
- `Song not found` in recommendation:
  - Use exact title from dataset
- XGBoost import failure:
  - Pipeline still runs with remaining models


# SongSage

**Discover genres and find your next favorite track.**

Live Demo: [https://gax4gvgkau3ey3dyza5uiz.streamlit.app](https://gax4gvgkau3ey3dyza5uiz.streamlit.app)  
Built by **Pragati**

SongSage is an end-to-end ML application that predicts music genres from audio features and recommends similar tracks. The project is built with production-style structure: modular pipelines, persisted artifacts, API endpoints, and a user-facing web app.

## Project Impact

- Built and deployed a complete ML product from dataset to live web app.
- Implemented multi-model genre classification and selected best model with weighted F1.
- Added a similarity-based recommendation engine for top-k track suggestions.
- Designed a clean Streamlit interface for non-technical users.
- Exposed model capabilities through FastAPI for service integration.

## Features

- Genre prediction from audio inputs
- Similar song recommendation by track name
- Model performance insights (Accuracy, Precision, Recall, F1)
- Deployment-ready artifact strategy (`models/deploy`)

## Tech Stack

- Python
- Pandas, NumPy
- scikit-learn, XGBoost
- FastAPI, Uvicorn
- Streamlit
- Matplotlib, Seaborn
- Joblib

## Dataset

- Spotify Tracks Dataset (Kaggle): [Link](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset)
- Target: `track_genre`
- Key features: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`

## Repository Structure

```text
spotify_ml_project/
├── api/
│   ├── app.py
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── deploy/
│   ├── saved_model/
│   └── saved_models/
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── recommendation_engine.py
│   ├── train_model.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Run Locally

```bash
cd /Users/pragatigodara/Documents/Playground/spotify_ml_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place dataset CSV in `data/raw/`, then train:

```bash
python main.py train
```

Run UI:

```bash
streamlit run api/streamlit_app.py
```

Run API:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /predict-genre` -> predicts genre from audio features
- `POST /recommend` -> returns top-k similar songs for an input song
- `POST /recommend-by-features` -> returns similar songs from custom feature vector

## Deployment Notes

- Streamlit entrypoint: `api/streamlit_app.py`
- Commit deploy artifacts in `models/deploy/`
- If artifacts are missing in cloud runtime, the app cannot load models

## Author

**Pragati**  
ML Engineer | Building practical, deployable machine learning systems.

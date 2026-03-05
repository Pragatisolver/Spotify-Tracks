# SongSage 🎧

**Predict song genres and discover similar tracks using Spotify audio features.**

SongSage is an **end-to-end machine learning application** built using Spotify audio feature data.  
It combines **genre classification** and **song recommendation** into a single interactive system.

The project is designed like a **production-style ML product**, featuring:

- modular training pipelines  
- saved model artifacts  
- FastAPI backend  
- interactive Streamlit frontend  

---

## Live Demo

🔗 **Try the App:**  
https://gax4gvgkau3ey3dyza5uiz.streamlit.app

---

## Core Features

- ### Genre Prediction
    Predict the **genre of a song** using Spotify audio features such as:
    `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`,  
    `instrumentalness`, `liveness`, `valence`, `tempo`.


- ### Song Recommendation
    Recommend **similar songs** using audio feature similarity.

    The system uses:
    `Standardized feature vectors` ,`Nearest Neighbors search` ,`cosine similarity ranking`.


- ### Model Insights 
     With core performance metrics and evaluation charts

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

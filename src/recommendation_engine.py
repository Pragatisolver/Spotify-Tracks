"""Song recommendation engine using nearest neighbors and cosine distance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import AUDIO_FEATURES, clean_dataset, find_dataset_file, load_dataset
from src.utils import save_joblib, setup_logger

logger = setup_logger(__name__)


@dataclass
class RecommenderArtifacts:
    """Serialized recommender payload."""

    model: NearestNeighbors
    scaler: StandardScaler
    feature_columns: list[str]
    catalog: pd.DataFrame


class SongRecommender:
    """Recommender wrapper with search by song name or feature vector."""

    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self.feature_columns = feature_columns or AUDIO_FEATURES
        self.scaler = StandardScaler()
        self.nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.catalog: pd.DataFrame | None = None
        self._feature_matrix: np.ndarray | None = None

    def fit(self, data: pd.DataFrame) -> None:
        """Fit scaler and nearest-neighbor index."""
        cols = [c for c in self.feature_columns if c in data.columns]
        if not cols:
            raise ValueError("No required recommendation features found in data")

        self.feature_columns = cols
        self.catalog = data.reset_index(drop=True)

        features = self.catalog[self.feature_columns]
        scaled = self.scaler.fit_transform(features)
        self.nn_model.fit(scaled)
        self._feature_matrix = scaled

    def recommend_by_song(self, song_name: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Recommend songs by existing song name."""
        if self.catalog is None or self._feature_matrix is None:
            raise RuntimeError("Recommender is not fitted")

        matches = self.catalog[self.catalog["track_name"].str.lower() == song_name.lower()]
        if matches.empty:
            raise ValueError(f"Song '{song_name}' not found")

        idx = matches.index[0]
        distances, indices = self.nn_model.kneighbors(
            self._feature_matrix[idx].reshape(1, -1),
            n_neighbors=min(top_k + 1, len(self.catalog)),
        )

        recs: list[dict[str, Any]] = []
        for distance, neighbor_idx in zip(distances[0], indices[0]):
            if neighbor_idx == idx:
                continue
            row = self.catalog.iloc[neighbor_idx]
            recs.append(
                {
                    "track_name": row.get("track_name", "unknown"),
                    "artists": row.get("artists", "unknown"),
                    "album_name": row.get("album_name", "unknown"),
                    "track_genre": row.get("track_genre", "unknown"),
                    "similarity_score": float(1 - distance),
                }
            )
            if len(recs) == top_k:
                break

        return recs

    def recommend_by_features(self, features: dict[str, float], top_k: int = 5) -> list[dict[str, Any]]:
        """Recommend songs from a user-provided audio feature vector."""
        if self.catalog is None:
            raise RuntimeError("Recommender is not fitted")

        row = [float(features.get(col, 0.0)) for col in self.feature_columns]
        scaled = self.scaler.transform([row])
        distances, indices = self.nn_model.kneighbors(
            scaled,
            n_neighbors=min(top_k, len(self.catalog)),
        )

        recs: list[dict[str, Any]] = []
        for distance, neighbor_idx in zip(distances[0], indices[0]):
            song = self.catalog.iloc[neighbor_idx]
            recs.append(
                {
                    "track_name": song.get("track_name", "unknown"),
                    "artists": song.get("artists", "unknown"),
                    "album_name": song.get("album_name", "unknown"),
                    "track_genre": song.get("track_genre", "unknown"),
                    "similarity_score": float(1 - distance),
                }
            )

        return recs

    def to_artifacts(self) -> RecommenderArtifacts:
        """Return serializable recommender artifacts."""
        if self.catalog is None:
            raise RuntimeError("Recommender is not fitted")
        return RecommenderArtifacts(
            model=self.nn_model,
            scaler=self.scaler,
            feature_columns=self.feature_columns,
            catalog=self.catalog,
        )

    @classmethod
    def from_artifacts(cls, artifacts: RecommenderArtifacts | dict[str, Any]) -> "SongRecommender":
        """Instantiate recommender from saved artifacts."""
        payload = artifacts if isinstance(artifacts, dict) else artifacts.__dict__
        recommender = cls(feature_columns=payload["feature_columns"])
        recommender.nn_model = payload["model"]
        recommender.scaler = payload["scaler"]
        recommender.catalog = payload["catalog"]
        recommender._feature_matrix = recommender.scaler.transform(
            recommender.catalog[recommender.feature_columns]
        )
        return recommender



def build_and_save_recommender(
    raw_data_path: Path | None = None,
    raw_dir: Path = Path("data/raw"),
    output_path: Path = Path("models/saved_models/song_recommender.joblib"),
) -> str:
    """Train recommendation model from cleaned dataset and save artifacts."""
    csv_path = raw_data_path if raw_data_path is not None else find_dataset_file(raw_dir)
    df = load_dataset(csv_path)
    cleaned = clean_dataset(df)

    recommender = SongRecommender()
    recommender.fit(cleaned)

    payload = recommender.to_artifacts().__dict__
    save_joblib(payload, output_path)
    logger.info("Recommender saved to %s", output_path)
    return str(output_path)

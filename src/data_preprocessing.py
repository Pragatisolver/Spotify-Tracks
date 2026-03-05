"""Data loading, cleaning, and splitting for Spotify genre classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ensure_directory, setup_logger

logger = setup_logger(__name__)

TARGET_COLUMN = "track_genre"
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
OPTIONAL_FEATURES = [
    "popularity",
    "duration_ms",
    "explicit",
    "key",
    "mode",
    "time_signature",
]
META_COLUMNS = ["track_name", "artists", "album_name", TARGET_COLUMN]


@dataclass
class PreprocessedData:
    """Container for preprocessed train/test data."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    full_dataframe: pd.DataFrame
    feature_columns: list[str]



def find_dataset_file(raw_dir: Path) -> Path:
    """Find a CSV dataset inside the raw data directory."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {raw_dir}. Download Kaggle dataset and place CSV in this folder."
        )
    return csv_files[0]



def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load dataset from CSV path."""
    logger.info("Loading dataset from %s", csv_path)
    return pd.read_csv(csv_path)



def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and remove duplicates."""
    data = df.copy()

    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    before = len(data)
    subset = [c for c in ["track_name", "artists", TARGET_COLUMN] if c in data.columns]
    data = data.drop_duplicates(subset=subset if subset else None)
    logger.info("Removed %s duplicate rows", before - len(data))

    if "explicit" in data.columns:
        explicit_map = {True: 1, False: 0, "True": 1, "False": 0, "yes": 1, "no": 0}
        data["explicit"] = data["explicit"].map(explicit_map).fillna(data["explicit"])

    for col in data.columns:
        if data[col].dtype == "object":
            mode = data[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            data[col] = data[col].fillna(fill_value)
        else:
            data[col] = data[col].fillna(data[col].median())

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' not found in dataset")

    return data



def _available_features(df: pd.DataFrame) -> list[str]:
    feature_candidates = AUDIO_FEATURES + OPTIONAL_FEATURES
    feature_columns = [col for col in feature_candidates if col in df.columns]
    if not feature_columns:
        raise ValueError("No expected feature columns found in dataset")
    return feature_columns



def split_data(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> PreprocessedData:
    """Split cleaned data into train/test datasets."""
    feature_columns = _available_features(data)

    X = data[feature_columns].copy()
    y = data[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        full_dataframe=data,
        feature_columns=feature_columns,
    )



def save_processed_data(data: pd.DataFrame, processed_dir: Path) -> Path:
    """Save cleaned dataframe for downstream use."""
    ensure_directory(processed_dir)
    out_path = processed_dir / "spotify_cleaned.csv"
    data.to_csv(out_path, index=False)
    logger.info("Saved cleaned data to %s", out_path)
    return out_path



def run_preprocessing(
    raw_data_path: Path | None = None,
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessedData:
    """Run full preprocessing pipeline and return split datasets."""
    csv_path = raw_data_path if raw_data_path is not None else find_dataset_file(raw_dir)
    data = load_dataset(csv_path)
    cleaned = clean_dataset(data)
    save_processed_data(cleaned, processed_dir)
    return split_data(cleaned, test_size=test_size, random_state=random_state)

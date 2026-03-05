"""Utility helpers for the Spotify ML project."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and return a configured logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def ensure_directory(path: os.PathLike[str] | str) -> Path:
    """Create a directory if it does not exist and return it as Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_joblib(obj: Any, file_path: os.PathLike[str] | str) -> Path:
    """Persist a Python object with joblib."""
    path = Path(file_path)
    ensure_directory(path.parent)
    # Compression keeps deployment artifacts much smaller.
    joblib.dump(obj, path, compress=3)
    return path


def load_joblib(file_path: os.PathLike[str] | str) -> Any:
    """Load a joblib artifact from disk."""
    return joblib.load(file_path)


def save_json(data: dict[str, Any], file_path: os.PathLike[str] | str) -> Path:
    """Persist a dictionary as a JSON file."""
    path = Path(file_path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path

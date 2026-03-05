"""Training pipeline for Spotify genre classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from src.data_preprocessing import run_preprocessing
from src.evaluate_model import (
    evaluate_classification,
    save_confusion_matrix_plot,
    save_model_comparison_plot,
)
from src.feature_engineering import build_features
from src.utils import ensure_directory, save_joblib, save_json, setup_logger

logger = setup_logger(__name__)

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # pragma: no cover
    HAS_XGBOOST = False



def get_models(random_state: int = 42) -> dict[str, Any]:
    """Create candidate models for comparison."""
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=2000, n_jobs=None),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "svm": SVC(kernel="rbf", probability=True, random_state=random_state),
    }

    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=350,
            learning_rate=0.08,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        logger.warning("xgboost is not available. Skipping XGBoost model.")

    return models



def maybe_tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    run_tuning: bool,
) -> RandomForestClassifier:
    """Optional hyperparameter tuning with GridSearchCV (bonus)."""
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    if not run_tuning:
        base_model.set_params(n_estimators=300)
        return base_model

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5],
    }

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info("Best RandomForest params: %s", search.best_params_)
    return search.best_estimator_



def train_pipeline(
    raw_data_path: Path | None = None,
    data_dir: Path = Path("data"),
    model_dir: Path = Path("models/saved_models"),
    test_size: float = 0.2,
    use_pca: bool = False,
    k_best: int | None = None,
    tune_hyperparameters: bool = False,
) -> dict[str, Any]:
    """Run full training workflow and save best model artifacts."""
    ensure_directory(model_dir)

    preprocessed = run_preprocessing(
        raw_data_path=raw_data_path,
        raw_dir=data_dir / "raw",
        processed_dir=data_dir / "processed",
        test_size=test_size,
    )

    engineered = build_features(
        X_train=preprocessed.X_train,
        X_test=preprocessed.X_test,
        y_train=preprocessed.y_train,
        y_test=preprocessed.y_test,
        k_best=k_best,
        use_pca=use_pca,
    )

    models = get_models()
    if "random_forest" in models:
        models["random_forest"] = maybe_tune_random_forest(
            engineered.X_train,
            engineered.y_train,
            run_tuning=tune_hyperparameters,
        )

    reports: dict[str, dict[str, Any]] = {}
    best_model_name = ""
    best_score = -1.0
    best_model: Any = None
    best_pred: pd.Series | None = None

    for name, model in models.items():
        logger.info("Training model: %s", name)
        model.fit(engineered.X_train, engineered.y_train)
        pred = pd.Series(model.predict(engineered.X_test), index=engineered.X_test.index)

        metrics = evaluate_classification(engineered.y_test, pred)
        reports[name] = metrics
        logger.info("%s metrics: accuracy=%.4f f1=%.4f", name, metrics["accuracy"], metrics["f1"])

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_model_name = name
            best_model = model
            best_pred = pred

    if best_model is None or best_pred is None:
        raise RuntimeError("No model was successfully trained")

    label_names = list(engineered.artifacts.label_encoder.classes_)

    confusion_path = model_dir / "best_model_confusion_matrix.png"
    save_confusion_matrix_plot(
        y_true=engineered.y_test,
        y_pred=best_pred,
        labels=label_names,
        output_path=confusion_path,
    )

    comparison_plot_path = model_dir / "model_comparison.png"
    save_model_comparison_plot(reports, comparison_plot_path)

    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": preprocessed.feature_columns,
        "selected_feature_names": engineered.artifacts.selected_feature_names,
        "scaler": engineered.artifacts.scaler,
        "pca": engineered.artifacts.pca,
        "label_encoder": engineered.artifacts.label_encoder,
        "metrics": reports[best_model_name],
    }

    model_path = save_joblib(artifact, model_dir / "best_genre_model.joblib")
    save_json(reports, model_dir / "all_model_metrics.json")

    logger.info("Best model: %s", best_model_name)
    logger.info("Model artifact saved: %s", model_path)

    return {
        "best_model": best_model_name,
        "best_model_path": str(model_path),
        "reports": reports,
        "comparison_plot": str(comparison_plot_path),
        "confusion_matrix_plot": str(confusion_path),
    }


def train_deploy_pipeline(
    raw_data_path: Path | None = None,
    data_dir: Path = Path("data"),
    deploy_dir: Path = Path("models/deploy"),
    test_size: float = 0.2,
    k_best: int = 10,
) -> dict[str, Any]:
    """Train a lightweight deployment-friendly classifier artifact."""
    ensure_directory(deploy_dir)

    preprocessed = run_preprocessing(
        raw_data_path=raw_data_path,
        raw_dir=data_dir / "raw",
        processed_dir=data_dir / "processed",
        test_size=test_size,
    )

    engineered = build_features(
        X_train=preprocessed.X_train,
        X_test=preprocessed.X_test,
        y_train=preprocessed.y_train,
        y_test=preprocessed.y_test,
        k_best=k_best,
        use_pca=False,
    )

    model = LogisticRegression(max_iter=2000, n_jobs=None)
    model.fit(engineered.X_train, engineered.y_train)
    pred = pd.Series(model.predict(engineered.X_test), index=engineered.X_test.index)
    metrics = evaluate_classification(engineered.y_test, pred)

    artifact = {
        "model": model,
        "model_name": "logistic_regression_deploy",
        "feature_columns": preprocessed.feature_columns,
        "selected_feature_names": engineered.artifacts.selected_feature_names,
        "scaler": engineered.artifacts.scaler,
        "pca": None,
        "label_encoder": engineered.artifacts.label_encoder,
        "metrics": metrics,
    }

    model_path = save_joblib(artifact, deploy_dir / "best_genre_model.joblib")
    save_json({"logistic_regression_deploy": metrics}, deploy_dir / "all_model_metrics.json")
    logger.info("Deployment model saved: %s", model_path)

    return {
        "deploy_model": "logistic_regression_deploy",
        "deploy_model_path": str(model_path),
        "metrics": metrics,
    }

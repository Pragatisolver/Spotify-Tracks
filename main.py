"""Main entrypoint for training and local recommendation inference."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.recommendation_engine import SongRecommender, build_and_save_recommender
from src.train_model import train_deploy_pipeline, train_pipeline
from src.utils import load_joblib, setup_logger

logger = setup_logger(__name__)


def resolve_recommender_path(root: Path) -> Path:
    """Resolve recommender path from supported model directories."""
    candidates = [
        root / "models" / "saved_models" / "song_recommender.joblib",
        root / "models" / "saved_model" / "song_recommender.joblib",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def sync_legacy_model_dir(root: Path) -> None:
    """Mirror artifacts to legacy models/saved_model path for compatibility."""
    source_dir = root / "models" / "saved_models"
    legacy_dir = root / "models" / "saved_model"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    for file_name in [
        "best_genre_model.joblib",
        "song_recommender.joblib",
        "all_model_metrics.json",
        "model_comparison.png",
        "best_model_confusion_matrix.png",
    ]:
        src = source_dir / file_name
        dst = legacy_dir / file_name
        if src.exists():
            shutil.copy2(src, dst)


def sync_deploy_dir(root: Path) -> None:
    """Sync recommender/plots to deploy directory."""
    deploy_dir = root / "models" / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    source_dir = root / "models" / "saved_models"

    for file_name in [
        "song_recommender.joblib",
        "all_model_metrics.json",
        "model_comparison.png",
        "best_model_confusion_matrix.png",
    ]:
        src = source_dir / file_name
        dst = deploy_dir / file_name
        if src.exists():
            shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(description="Spotify ML Project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train classification model and recommender")
    train_parser.add_argument("--data-csv", type=str, default=None, help="Path to raw Spotify CSV")
    train_parser.add_argument("--use-pca", action="store_true", help="Use PCA dimensionality reduction")
    train_parser.add_argument("--k-best", type=int, default=None, help="Top-k feature selection")
    train_parser.add_argument(
        "--tune",
        action="store_true",
        help="Run optional GridSearchCV tuning for RandomForest",
    )

    rec_parser = subparsers.add_parser("recommend", help="Get song recommendations")
    rec_parser.add_argument("--song", type=str, required=True, help="Song name")
    rec_parser.add_argument("--top-k", type=int, default=5)

    return parser.parse_args()


def run_train(args: argparse.Namespace) -> None:
    """Train and persist all artifacts."""
    root = Path(__file__).resolve().parent
    data_csv = Path(args.data_csv) if args.data_csv else None

    summary = train_pipeline(
        raw_data_path=data_csv,
        data_dir=root / "data",
        model_dir=root / "models" / "saved_models",
        use_pca=args.use_pca,
        k_best=args.k_best,
        tune_hyperparameters=args.tune,
    )

    recommender_path = build_and_save_recommender(
        raw_data_path=data_csv,
        raw_dir=root / "data" / "raw",
        output_path=root / "models" / "saved_models" / "song_recommender.joblib",
    )
    deploy_summary = train_deploy_pipeline(
        raw_data_path=data_csv,
        data_dir=root / "data",
        deploy_dir=root / "models" / "deploy",
    )
    sync_legacy_model_dir(root)
    sync_deploy_dir(root)

    logger.info("Training completed. Best model: %s", summary["best_model"])
    logger.info("Recommender saved at: %s", recommender_path)
    logger.info("Deploy-friendly model: %s", deploy_summary["deploy_model_path"])


def run_recommend(args: argparse.Namespace) -> None:
    """Load recommender artifact and print recommendations."""
    root = Path(__file__).resolve().parent
    artifact_path = resolve_recommender_path(root)

    if not artifact_path.exists():
        raise FileNotFoundError(f"Recommender artifact not found at {artifact_path}. Run train first.")

    payload = load_joblib(artifact_path)
    recommender = SongRecommender.from_artifacts(payload)

    recs = recommender.recommend_by_song(args.song, top_k=args.top_k)
    logger.info("Recommendations for '%s':", args.song)
    for idx, rec in enumerate(recs, start=1):
        logger.info(
            "%d. %s - %s (genre=%s, score=%.4f)",
            idx,
            rec["track_name"],
            rec["artists"],
            rec["track_genre"],
            rec["similarity_score"],
        )


def main() -> None:
    """Program entrypoint."""
    args = parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "recommend":
        run_recommend(args)


if __name__ == "__main__":
    main()

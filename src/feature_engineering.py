"""Feature engineering utilities: scaling, selection, and PCA."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class FeatureArtifacts:
    """Persistable artifacts from feature engineering."""

    scaler: StandardScaler
    label_encoder: LabelEncoder
    selected_feature_names: list[str]
    pca: PCA | None


@dataclass
class EngineeredData:
    """Container for transformed train/test datasets."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    artifacts: FeatureArtifacts



def encode_labels(y_train: pd.Series, y_test: pd.Series) -> tuple[pd.Series, pd.Series, LabelEncoder]:
    """Encode string genre labels into numeric classes."""
    encoder = LabelEncoder()
    y_train_enc = pd.Series(encoder.fit_transform(y_train), index=y_train.index)
    y_test_enc = pd.Series(encoder.transform(y_test), index=y_test.index)
    return y_train_enc, y_test_enc, encoder



def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    k_best: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Select top-k informative features using mutual information."""
    if k_best is None or k_best <= 0 or k_best >= X_train.shape[1]:
        return X_train.copy(), X_test.copy(), list(X_train.columns)

    selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    mask = selector.get_support()
    selected_names = list(X_train.columns[mask])

    return (
        pd.DataFrame(X_train_selected, index=X_train.index, columns=selected_names),
        pd.DataFrame(X_test_selected, index=X_test.index, columns=selected_names),
        selected_names,
    )



def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardize numerical features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns),
        pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns),
        scaler,
    )



def apply_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: float | int = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """Apply PCA dimensionality reduction."""
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    cols = [f"pc_{idx + 1}" for idx in range(X_train_pca.shape[1])]
    return (
        pd.DataFrame(X_train_pca, index=X_train.index, columns=cols),
        pd.DataFrame(X_test_pca, index=X_test.index, columns=cols),
        pca,
    )



def build_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    k_best: int | None = None,
    use_pca: bool = False,
) -> EngineeredData:
    """Run complete feature engineering flow."""
    y_train_enc, y_test_enc, label_encoder = encode_labels(y_train, y_test)

    X_train_sel, X_test_sel, selected_names = select_features(
        X_train=X_train,
        y_train=y_train_enc,
        X_test=X_test,
        k_best=k_best,
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_sel, X_test_sel)

    pca_model: PCA | None = None
    if use_pca:
        X_train_scaled, X_test_scaled, pca_model = apply_pca(X_train_scaled, X_test_scaled)

    artifacts = FeatureArtifacts(
        scaler=scaler,
        label_encoder=label_encoder,
        selected_feature_names=selected_names,
        pca=pca_model,
    )

    return EngineeredData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train_enc,
        y_test=y_test_enc,
        artifacts=artifacts,
    )

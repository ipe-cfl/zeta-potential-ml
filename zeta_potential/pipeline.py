import math
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import EPSILON_BI_MONO, EPSILON_SIGMA, FEATURE_NAMES, MODEL_PATH
from .validation import validate_dataframe, validate_row

_model: Any | None = None


def load_model() -> Any:
    global _model  # noqa: PLW0603
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def _inject_log_features(
    raw_features: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Return a copy of raw_features with log(bi/mono) and log(sigma) set.

    The caller can supply either:
    - log(bi/mono) and log(sigma) directly, or
    - raw bi, mono, and sigma values.

    In the second case we compute the logs using the configured epsilons.
    """
    features = dict(raw_features)
    errors: list[str] = []

    has_logged = "log(bi/mono)" in features and "log(sigma)" in features
    has_raw = "bi" in features and "mono" in features and "sigma" in features

    if has_logged and has_raw:
        errors.append(
            "Provide either logged ionic features ('log(bi/mono)', 'log(sigma)') "
            "or raw values ('bi', 'mono', 'sigma'), not both."
        )
        return features, errors

    if not has_logged:
        if not has_raw:
            errors.append(
                "Missing ionic features: provide either 'log(bi/mono)' and 'log(sigma)' "
                "or raw 'bi', 'mono', 'sigma'."
            )
            return features, errors

        # Parse raw values and compute logs
        try:
            bi = float(features["bi"])
            mono = float(features["mono"])
            sigma = float(features["sigma"])
        except (TypeError, ValueError) as exc:
            errors.append(
                "Raw ionic features 'bi', 'mono' and 'sigma' must be numeric "
                f"to compute their logarithms: {exc}"
            )
            return features, errors

        # Basic sanity checks for the log arguments
        if bi + EPSILON_BI_MONO <= 0.0:
            errors.append("bi + epsilon must be positive for log(bi/mono) computation.")
        if mono + EPSILON_BI_MONO <= 0.0:
            errors.append("mono + epsilon must be positive for log(bi/mono) computation.")
        if sigma + EPSILON_SIGMA <= 0.0:
            errors.append("sigma + epsilon must be positive for log(sigma) computation.")
        if errors:
            return features, errors

        try:
            features["log(bi/mono)"] = math.log((bi + EPSILON_BI_MONO) / (mono + EPSILON_BI_MONO))
            features["log(sigma)"] = math.log(sigma + EPSILON_SIGMA)
        except ValueError as exc:
            errors.append(f"Could not compute log features: {exc}")

    # If has_logged is True we just pass through.
    return features, errors


def _prepare_bulk_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Ensure df has all model feature columns, computing log features if needed.

    The input dataframe can either include:
      - 'log(bi/mono)' and 'log(sigma)' directly, or
      - raw 'bi', 'mono', 'sigma' columns instead of the two log columns.

    Returns (df_features, errors). If errors is non-empty, df_features is the
    original dataframe and the caller should not run the model.
    """
    errors: list[str] = []

    cols = set(df.columns)
    has_logged = {"log(bi/mono)", "log(sigma)"}.issubset(cols)
    has_raw = {"bi", "mono", "sigma"}.issubset(cols)

    if has_logged and has_raw:
        errors.append(
            "For bulk predictions, provide either logged ionic features "
            "('log(bi/mono)', 'log(sigma)') or raw values ('bi', 'mono', 'sigma'), not both."
        )
        return df, errors

    if not has_logged and not has_raw:
        errors.append(
            "For bulk predictions, dataframe must contain either the logged features "
            "('log(bi/mono)', 'log(sigma)') or raw ionic columns ('bi', 'mono', 'sigma')."
        )
        return df, errors

    df_work = df.copy()

    if not has_logged:
        # Compute logs from raw columns.
        try:
            bi = df_work["bi"].astype(float)
            mono = df_work["mono"].astype(float)
            sigma = df_work["sigma"].astype(float)
        except (TypeError, ValueError) as exc:
            errors.append(
                "Raw ionic columns 'bi', 'mono' and 'sigma' must be numeric "
                f"to compute their logarithms: {exc}"
            )
            return df, errors

        bi_shifted = bi + EPSILON_BI_MONO
        mono_shifted = mono + EPSILON_BI_MONO
        sigma_shifted = sigma + EPSILON_SIGMA

        invalid_mask = (bi_shifted <= 0.0) | (mono_shifted <= 0.0) | (sigma_shifted <= 0.0)
        if invalid_mask.any():
            bad_indices = list(df_work.index[invalid_mask])
            errors.append(
                "Cannot compute logarithms for rows with indices "
                + ", ".join(str(i) for i in bad_indices)
                + "; check that bi + epsilon, mono + epsilon and sigma + epsilon are positive."
            )
            return df, errors

        df_work["log(bi/mono)"] = np.log(bi_shifted / mono_shifted)
        df_work["log(sigma)"] = np.log(sigma_shifted)

    # At this point, log(bi/mono) and log(sigma) exist.
    missing = [name for name in FEATURE_NAMES if name not in df_work.columns]
    if missing:
        errors.append("Missing required feature columns for prediction: " + ", ".join(missing))
        return df, errors

    return df_work[FEATURE_NAMES].copy(), errors


def predict_single(
    features: dict[str, Any],
) -> tuple[float | None, list[str], list[str]]:
    """Predict zeta potential (mV) for a single feature dict.

    The caller can either pass:
      - the logged features expected by the model (keys in FEATURE_NAMES), or
      - raw ionic inputs 'bi', 'mono', 'sigma' instead of 'log(bi/mono)' and
        'log(sigma)'. In that case the logarithms are computed internally.

    Returns (prediction_mV, errors, warnings).
    If errors is non-empty, prediction_mV will be None.
    """
    # Compute log(bi/mono) and log(sigma) if only raw inputs are provided.
    with_logs, pre_errors = _inject_log_features(features)

    clean, errors, warnings = validate_row(with_logs)
    all_errors = [*pre_errors, *errors]

    if all_errors:
        return None, all_errors, warnings

    model = load_model()
    X = pd.DataFrame([[clean[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)  # noqa: N806
    y_pred = model.predict(X.to_numpy())
    pred = float(y_pred[0])
    return pred, all_errors, warnings


def predict_bulk(
    df: pd.DataFrame,
    validate: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Run bulk predictions on a dataframe.

    The dataframe may contain either:
      - the six feature columns in FEATURE_NAMES (including 'log(bi/mono)' and 'log(sigma)'), or
      - raw ionic columns 'bi', 'mono', 'sigma' instead of the two log columns.

    Any extra columns are ignored.

    Returns (df_with_pred, messages) where df_with_pred has a 'predicted_zeta_mV' column.
    """
    messages: list[str] = []

    # Compute log(bi/mono) and log(sigma) if needed and ensure the six model features exist.
    df_features, prep_errors = _prepare_bulk_features(df)

    if prep_errors:
        # If preparation itself fails, surface the errors and do not call the model.
        return df.copy(), prep_errors

    if validate:
        df_clean, val_messages = validate_dataframe(df_features)
        messages.extend(val_messages)
    else:
        df_clean = df_features.copy()

    model = load_model()
    X = df_clean.to_numpy()  # noqa: N806
    y_pred = model.predict(X)
    df_out = df_clean.copy()
    df_out["predicted_zeta_mV"] = y_pred
    return df_out, messages

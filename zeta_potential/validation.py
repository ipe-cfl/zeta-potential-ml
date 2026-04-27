from typing import Any

import pandas as pd

from .config import FEATURE_BOUNDS, FEATURE_NAMES


def validate_row(raw: dict[str, Any]) -> tuple[dict[str, float], list[str], list[str]]:
    """Validate a single row of features.

    Returns (clean_row, errors, warnings).

    - errors: structural issues (missing/non-numeric features, invalid categories)
    - warnings: features whose values are outside model-defined bounds
    """
    clean: dict[str, float] = {}
    errors: list[str] = []
    out_of_bounds_features: set[str] = set()

    for name in FEATURE_NAMES:
        if name not in raw:
            errors.append(f"Missing feature: {name}")
            continue

        try:
            value = float(raw[name])
        except (TypeError, ValueError):
            errors.append(f"Feature {name} must be numeric, got {raw[name]!r}")
            continue

        bounds = FEATURE_BOUNDS[name]

        # Any violation of hard bounds or training range → warning (feature name only)
        out_of_range = False

        if bounds.hard_min is not None and value < bounds.hard_min:
            out_of_range = True

        if bounds.hard_max is not None and value > bounds.hard_max:
            out_of_range = True

        if value < bounds.train_min or value > bounds.train_max:
            out_of_range = True

        if out_of_range:
            # Only track the feature name; details are not exposed to the user
            out_of_bounds_features.add(name)

        clean[name] = value

    # Special case for Clay Rich Silicate
    if "Clay Rich Silicate" in clean:
        v = round(clean["Clay Rich Silicate"])
        if v not in (0, 1):
            errors.append(
                f"Clay Rich Silicate must be 0 or 1, got {clean['Clay Rich Silicate']:.4g}"
            )
        clean["Clay Rich Silicate"] = float(v)

    warnings = sorted(out_of_bounds_features)
    return clean, errors, warnings


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate a bulk dataframe row by row.

    Returns (clean_df, messages). The dataframe is always returned with
    exactly the FEATURE_NAMES columns in the right order.
    """
    rows: list[dict[str, float]] = []
    messages: list[str] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        clean, errors, warnings = validate_row(row.to_dict())
        if errors:
            messages.append(f"Row {idx}: ERROR: " + "; ".join(errors))
        if warnings:
            feature_list = ", ".join(warnings)
            messages.append(
                f"Row {idx}: WARNING: features outside model-defined bounds: {feature_list}"
            )
        rows.append(clean)

    clean_df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return clean_df, messages

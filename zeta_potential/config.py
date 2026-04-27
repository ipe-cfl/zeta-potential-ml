from dataclasses import dataclass
from pathlib import Path

# Epsilon values for logarithmic features
EPSILON_BI_MONO: float = 1e-6
EPSILON_SIGMA: float = 1e-7

# Features in the exact order expected by the model
FEATURE_NAMES: list[str] = [
    "SiO2(%)",
    "pH",
    "Temperature",
    "log(bi/mono)",
    "log(sigma)",
    "Clay Rich Silicate",
]

# Path to the serialized XGBoost model (joblib)
MODEL_PATH: Path = Path(__file__).resolve().parents[1] / "models" / "xgb_zeta_model.joblib"


@dataclass
class Bounds:
    train_min: float
    train_max: float
    hard_min: float | None = None
    hard_max: float | None = None
    unit: str | None = None


FEATURE_BOUNDS: dict[str, Bounds] = {
    "SiO2(%)": Bounds(
        train_min=0.0,
        train_max=100.0,
        hard_min=0.0,
        hard_max=100.0,
        unit="wt%",
    ),
    "pH": Bounds(
        train_min=1.0,
        train_max=12.987705,
        hard_min=0.0,
        hard_max=14.0,
        unit="pH units",
    ),
    "Temperature": Bounds(
        train_min=15.0,
        train_max=80.0,
        hard_min=0.0,
        hard_max=200.0,
        unit="°C",
    ),
    "log(bi/mono)": Bounds(
        train_min=-14.704316,
        train_max=13.864586,
        hard_min=-20.0,
        hard_max=20.0,
        unit="dimensionless (ln)",
    ),
    "log(sigma)": Bounds(
        train_min=-16.118096,
        train_max=-3.074377,
        hard_min=-25.0,
        hard_max=0.0,
        unit="dimensionless (ln)",
    ),
    "Clay Rich Silicate": Bounds(
        train_min=0.0,
        train_max=1.0,
        hard_min=0.0,
        hard_max=1.0,
        unit="0 or 1",
    ),
}

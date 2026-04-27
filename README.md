# Zeta Potential Predictor

A lightweight Python project for predicting zeta potential in millivolts (mV) with a trained XGBoost model. The repository includes a local Gradio interface for single-sample predictions and a Python API for single or bulk inference.

This README is written for a normal GitHub repository, not for a Hugging Face Space. It intentionally does not include Hugging Face Spaces front matter.


[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-yellow)](https://huggingface.co/spaces/ipe-cfl/zeta-potential-ml-demo)

## Live demo

You can try the model without installing anything by using the hosted Hugging Face Space:

[Open the Zeta Potential ML Demo](https://huggingface.co/spaces/ipe-cfl/zeta-potential-ml-demo)

Use the demo for quick, interactive predictions. Use this GitHub repository when you want to inspect the source code, run the app locally, or use the Python prediction API in your own workflow.

## Features

- Interactive local web UI built with Gradio.
- Single-sample prediction through `predict_single`.
- Bulk prediction from a `pandas.DataFrame` through `predict_bulk`.
- Automatic conversion of raw ionic inputs into the log-space features expected by the model.
- Input validation with warnings for values outside the model's training bounds.

## Repository layout

```text
.
├── app.py                         # Gradio web application
├── requirements.txt               # Core model/inference dependencies
├── models/
│   └── xgb_zeta_model.joblib       # Serialized trained XGBoost model
└── zeta_potential/
    ├── __init__.py
    ├── config.py                   # Feature order, model path, constants, bounds
    ├── pipeline.py                 # Model loading and prediction functions
    └── validation.py               # Single-row and bulk validation helpers
```

## Model inputs

The trained model expects the following six features, in this exact order:

| Feature | Description |
|---|---|
| `SiO2(%)` | SiO₂ content in wt% |
| `pH` | Solution pH |
| `Temperature` | Temperature in °C |
| `log(bi/mono)` | Natural log of the bi-valent to monovalent concentration ratio |
| `log(sigma)` | Natural log of surface charge density |
| `Clay Rich Silicate` | Binary indicator, `0` or `1` |

For convenience, the prediction pipeline can accept either the logged ionic features directly or the raw columns `bi`, `mono`, and `sigma`. When raw values are supplied, the code computes:

```python
log(bi/mono) = log((bi + 1e-6) / (mono + 1e-6))
log(sigma) = log(sigma + 1e-7)
```

Do not provide both raw ionic values and logged ionic values in the same prediction call.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone <your-repository-url>
cd <your-repository-name>

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

To run the web UI, Gradio is also required. The current zip imports `gradio` in `app.py`, but `gradio` is not listed in `requirements.txt`, so install it separately or add it to `requirements.txt`:

```bash
python -m pip install gradio
```

Recommended `requirements.txt` entry for the UI:

```text
gradio
```

## Run the local web app

```bash
python app.py
```

Gradio will start a local server and print a URL such as:

```text
http://127.0.0.1:7860
```

Open that URL in your browser, enter the input values, and click **Predict zeta potential**.

The UI currently exposes these inputs:

- `SiO2(%)`
- `pH`
- `Temperature`
- `bi`
- `mono`
- `sigma`

`Clay Rich Silicate` is fixed to `0` in the current UI implementation through a Gradio state value. If you need to expose it to users, replace the internal `gr.State(0)` with a visible radio button, checkbox, or dropdown.

## Python API: single prediction

```python
from zeta_potential.pipeline import predict_single

features = {
    "SiO2(%)": 44.52,
    "pH": 7.08,
    "Temperature": 27.05,
    "bi": 0.0652,
    "mono": 1.0,
    "sigma": 0.000248,
    "Clay Rich Silicate": 0,
}

prediction_mV, errors, warnings = predict_single(features)

if errors:
    print("Prediction failed:")
    for error in errors:
        print(f"- {error}")
else:
    print(f"Predicted zeta potential: {prediction_mV:.3f} mV")

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- Feature outside model-defined bounds: {warning}")
```

You can also pass logged ionic features directly:

```python
features = {
    "SiO2(%)": 44.52,
    "pH": 7.08,
    "Temperature": 27.05,
    "log(bi/mono)": -2.730,
    "log(sigma)": -8.300,
    "Clay Rich Silicate": 0,
}
```

## Python API: bulk prediction

```python
import pandas as pd

from zeta_potential.pipeline import predict_bulk

input_df = pd.read_csv("samples.csv")
output_df, messages = predict_bulk(input_df)

for message in messages:
    print(message)

output_df.to_csv("samples_with_zeta_predictions.csv", index=False)
```

Bulk input can use either raw ionic columns:

```text
SiO2(%),pH,Temperature,bi,mono,sigma,Clay Rich Silicate
44.52,7.08,27.05,0.0652,1.0,0.000248,0
```

or logged ionic columns:

```text
SiO2(%),pH,Temperature,log(bi/mono),log(sigma),Clay Rich Silicate
44.52,7.08,27.05,-2.730,-8.300,0
```

The current `predict_bulk` implementation returns the model feature columns plus `predicted_zeta_mV`. Extra input columns are ignored and are not preserved in the returned dataframe.

## Validation behavior

The pipeline validates inputs before inference.

Errors stop prediction and are returned when:

- Required features are missing.
- Feature values are not numeric.
- `Clay Rich Silicate` is not `0` or `1`.
- Raw `bi`, `mono`, or `sigma` values would make the logarithm arguments non-positive.
- Both raw ionic columns and logged ionic columns are provided at the same time.

Warnings are returned when feature values fall outside the model-defined training bounds. Predictions are still produced in this case, but they should be interpreted carefully.

## Model file

The model is loaded from:

```text
models/xgb_zeta_model.joblib
```

Keep this file in the same relative location unless you also update `MODEL_PATH` in `zeta_potential/config.py`.

## Development notes

Run a quick smoke test from the repository root:

```bash
python - <<'PY'
from zeta_potential.pipeline import predict_single

features = {
    "SiO2(%)": 44.52,
    "pH": 7.08,
    "Temperature": 27.05,
    "bi": 0.0652,
    "mono": 1.0,
    "sigma": 0.000248,
    "Clay Rich Silicate": 0,
}

prediction, errors, warnings = predict_single(features)
print("prediction:", prediction)
print("errors:", errors)
print("warnings:", warnings)
PY
```

## Limitations

This project provides machine-learning inference, not a physical simulation. Predictions are most reliable for inputs similar to the model's training domain. Values outside the training bounds may produce extrapolated results and should be reviewed carefully.

The concentration and surface charge density units must match the units used during model training.

## License

The original Hugging Face Space metadata declared the project license as Apache-2.0. If publishing this project on GitHub, add a `LICENSE` file containing the Apache License 2.0 text or replace this section with the correct license for your repository.

## Citation

If this model or repository is used in a research article, cite the GitHub repository and include the model/data provenance required by your project or publication venue.

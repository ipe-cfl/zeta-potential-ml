import gradio as gr

from zeta_potential.pipeline import predict_single


def predict_ui(
    sio2: float,
    ph: float,
    temp: float,
    bi: float,
    mono: float,
    sigma: float,
    clay: int,
) -> tuple[float | None, str]:
    # Pass raw bi, mono, sigma; pipeline will compute the logs internally.
    features = {
        "SiO2(%)": sio2,
        "pH": ph,
        "Temperature": temp,
        "bi": bi,
        "mono": mono,
        "sigma": sigma,
        "Clay Rich Silicate": int(clay),
    }
    pred, errors, warnings = predict_single(features)

    if errors:
        msg_lines: list[str] = ["❌ Input error(s):"]
        msg_lines.extend(f"- {e}" for e in errors)
        if warnings:
            feature_list = ", ".join(warnings)
            msg_lines.append(
                "\n⚠️ Additionally, these features are outside the model's training bounds: "
                f"{feature_list}."
            )
        return None, "\n".join(msg_lines)

    msg_lines = ["✅ Inference completed."]
    if warnings:
        feature_list = ", ".join(warnings)
        msg_lines.append(
            f"\n⚠️ The following features are outside the model's training bounds: {feature_list}."
        )

    msg_lines.append(
        "\nAll predictions are in millivolts (mV). Use with caution when any feature is outside "
        "the training bounds."
    )

    return pred, "\n".join(msg_lines)


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Zeta Potential Predictor (XGBoost)

        Predicts zeta potential (mV) from the following inputs:

        1. SiO₂ content, wt% (`SiO2(%)`)
        2. pH
        3. Temperature (°C)
        4. Bi-valent cation concentration (`bi`)
        5. Monovalent cation concentration (`mono`)
        6. Surface charge density (`sigma`)

        Internally, the model uses:
        - `log(bi/mono) = log((bi + 1e-6) / (mono + 1e-6))`
        - `log(sigma) = log(sigma + 1e-7)`

        Inputs are validated against the training data ranges in this log space.
        Values outside the training range are allowed but trigger warnings.
        Physically impossible values are rejected.
        """
    )

    with gr.Row():
        with gr.Column():
            sio2_input = gr.Slider(
                minimum=0.0,
                maximum=100.0,
                value=44.52,
                step=0.1,
                label="SiO₂ content (SiO2(%), wt%)",
                info="Training range: 0-100 wt%",
            )
            ph_input = gr.Slider(
                minimum=0.0,
                maximum=14.0,
                value=7.08,
                step=0.01,
                label="pH",
                info="Training range: ~1.0-13.0",
            )
            temp_input = gr.Slider(
                minimum=0.0,
                maximum=100.0,
                value=27.05,
                step=0.5,
                label="Temperature (°C)",
                info="Training range: 15-80 °C",
            )
        with gr.Column():
            bi_input = gr.Number(
                value=0.0652,
                label="bi concentration",
                info=(
                    "Bi-valent ion concentration used to compute log((bi + 1e-6)/(mono + 1e-6)). "
                    "Units must be consistent with training data."
                ),
            )
            mono_input = gr.Number(
                value=1.0,
                label="mono concentration",
                info="Monovalent ion concentration used to compute log((bi + 1e-6)/(mono + 1e-6)).",
            )
            sigma_input = gr.Number(
                value=0.000248,
                label="sigma",
                info=(
                    "Surface charge density used to compute log(sigma + 1e-7). Units must be "
                    "consistent with training data."
                ),
            )

    predict_button = gr.Button("Predict zeta potential")

    with gr.Row():
        pred_output = gr.Number(
            label="Predicted zeta potential (mV)",
            precision=3,
        )
        message_output = gr.Markdown()

    clay_state = gr.State(0)

    predict_button.click(
        fn=predict_ui,
        inputs=[
            sio2_input,
            ph_input,
            temp_input,
            bi_input,
            mono_input,
            sigma_input,
            clay_state,
        ],
        outputs=[pred_output, message_output],
        api_name="predict",  # so users can call this via gradio_client
    )

    gr.Markdown(
        """
        ### Bulk predictions

        This interface performs one prediction at a time.

        For bulk predictions, use the Python API:

        ```python
        import pandas as pd
        from zeta_potential.pipeline import predict_bulk

        # Your CSV can either contain:
        # - the logged features 'log(bi/mono)' and 'log(sigma)', or
        # - raw columns 'bi', 'mono', 'sigma' (logs will be computed internally).

        df = pd.read_csv("my_bulk_samples.csv")
        df_pred, messages = predict_bulk(df)
        df_pred.to_csv("my_bulk_samples_with_zeta.csv", index=False)

        for m in messages:
            print(m)
        ```
        """
    )

if __name__ == "__main__":
    demo.launch()

from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from telco_customer_churn_prediction.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_df.parquet",
    model_path: Path = MODELS_DIR / "telco_churn_xgboost_v1.pkl",
    output_path: Path = DATA_DIR / "predictions.csv",
):
    """
    Run inference using the trained XGBoost model.
    """
    logger.info(f"Loading features from {features_path}...")
    if not features_path.exists():
        logger.error("Features file not found!")
        raise FileNotFoundError

    # Load Data
    df = pd.read_parquet(features_path)

    # We need the Customer IDs to know WHO to call, but the model only wants features.
    # Assuming 'customer_id' and 'churn' are in the parquet but need to be dropped for prediction
    ids = df["customer_id"]

    # Define the exact features the model expects (Same as train.py)
    # Ideally, this list is stored in a config file to avoid duplication
    MODEL_FEATURES = [
        "trend_data_w4_vs_w1",
        "data_gini_coefficient",
        "trend_spend_w4_vs_w1",
        "trend_data_w3_vs_w1",
        "data_volatility_shift",
        "peak_spend_week",
        "spend_volatility_shift",
        "lowest_data_week",
        "trend_spend_w2_vs_w1",
        "peak_data_week",
        "trend_spend_w3_vs_w1",
        "trend_data_w2_vs_w1",
        "ratio_min_daily_data_to_avg",
        "pct_video_w4",
        "spend_consistency_score",
        "pct_messaging_w4",
        "pct_messaging_w3",
        "pct_video_w3",
        "pct_messaging_w2",
        "ratio_min_daily_spend_to_avg",
        "pct_messaging_w1",
        "pct_video_w2",
        "pct_video_w1",
    ]

    X = df[MODEL_FEATURES]

    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    logger.info("Running inference...")
    # Get probabilities (Risk Score)
    probs = model.predict_proba(X)[:, 1]

    # Create Results DataFrame
    results = pd.DataFrame({"customer_id": ids, "churn_probability": probs})

    # Sort by risk (Highest risk at top)
    results = results.sort_values("churn_probability", ascending=False)

    logger.info(f"Saving predictions to {output_path}...")
    results.to_csv(output_path, index=False)
    logger.success("Inference complete.")


if __name__ == "__main__":
    app()

from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import typer
from xgboost import XGBClassifier

from telco_customer_churn_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

# --- CONFIGURATION ---
# The best features selected during the EDA (Notebook 2.0 / 3.0)
SELECTED_FEATURES = [
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

# The best parameters found via Optuna (Notebook 3.0, Page 15)
BEST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
    "random_state": 42,
    "n_estimators": 2000,
    "learning_rate": 0.01006,
    "scale_pos_weight": 4.498,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.812,
    "colsample_bytree": 0.821,
    "gamma": 3.148,
    "reg_alpha": 0.488,
    "reg_lambda": 8.515,
}


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train_df.parquet",
    model_output_path: Path = MODELS_DIR / "telco_churn_xgboost_v1.pkl",
):
    """
    Trains the XGBoost Churn Classifier using pre-optimized hyperparameters.
    """
    logger.info("Loading training data...")
    if not train_path.exists():
        logger.error(f"{train_path} not found. Run 'make features' first.")
        raise FileNotFoundError

    train_df = pd.read_parquet(train_path)

    # Split X and y
    X_train = train_df[SELECTED_FEATURES]
    y_train = train_df["churn"]

    logger.info(f"Training XGBoost Model with {len(SELECTED_FEATURES)} features...")
    logger.info(f"Parameters: {BEST_PARAMS}")

    # Initialize Model
    xgb_model = XGBClassifier(**BEST_PARAMS)

    # Calibrate Model (Isotonic) - As per Notebook 3.0
    # This improves the accuracy of the probability output (predict_proba)
    final_model = CalibratedClassifierCV(estimator=xgb_model, method="isotonic", cv=5, n_jobs=-1)

    # Fit
    final_model.fit(X_train, y_train)
    logger.success("Model training complete.")

    # Save
    logger.info(f"Saving model to {model_output_path}...")
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_output_path)
    logger.success("Model serialized and saved.")


if __name__ == "__main__":
    app()

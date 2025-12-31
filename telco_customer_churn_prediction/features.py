from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from telco_customer_churn_prediction.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


# --- HELPER FUNCTIONS ---
def safe_div(a, b):
    """Vectorized safe division to avoid ZeroDivisionError."""
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def vectorized_gini(array_like):
    """Calculate Gini coefficient for a matrix (Customers x Days)."""
    # Ensure numpy array and sort row-wise
    arr = np.sort(array_like.values, axis=1)
    n = arr.shape[1]
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * arr, axis=1)) / (n * np.sum(arr, axis=1))


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "combined_usage.parquet",
    output_train_path: Path = PROCESSED_DATA_DIR / "train_df.parquet",
    output_test_path: Path = PROCESSED_DATA_DIR / "test_df.parquet",
):
    """
    Generates robust behavioral features (Trends, Gini, Cliffs) from usage data.
    """
    logger.info("--- Starting Feature Engineering ---")

    # 1. Load Data
    if not input_path.exists():
        logger.error(f"Input file {input_path} not found. Run 'make data' first.")
        raise FileNotFoundError

    combined_usage_df = pd.read_parquet(input_path)

    # 2. Define Windows & Categories
    W1_COLS = [f"Day_{i}" for i in range(1, 8)]
    W2_COLS = [f"Day_{i}" for i in range(8, 15)]
    W3_COLS = [f"Day_{i}" for i in range(15, 22)]
    W4_COLS = [f"Day_{i}" for i in range(22, 29)]
    ALL_OBS_COLS = W1_COLS + W2_COLS + W3_COLS + W4_COLS

    DATA_APPS = [
        "usage_app_facebook_daily",
        "usage_app_youtube_daily",
        "usage_app_tiktok_daily",
        "usage_app_whatsapp_daily",
        "usage_app_helakuru_daily",
        "usage_app_other",
    ]
    SPEND_TYPES = ["usage_pack_data", "usage_pack_vas"]
    VIDEO_APPS = ["usage_app_youtube_daily", "usage_app_tiktok_daily"]
    MSG_APPS = ["usage_app_whatsapp_daily"]

    # 3. Create Aggregates (Pivot)
    logger.info("Aggregating daily usage...")
    # Filter to relevant types
    df = combined_usage_df[combined_usage_df["usage_type"].isin(DATA_APPS + SPEND_TYPES)].copy()

    # Group and Sum
    data_df = df[df["usage_type"].isin(DATA_APPS)].groupby("customer_id")[ALL_OBS_COLS].sum()
    spend_df = df[df["usage_type"].isin(SPEND_TYPES)].groupby("customer_id")[ALL_OBS_COLS].sum()
    video_df = df[df["usage_type"].isin(VIDEO_APPS)].groupby("customer_id")[ALL_OBS_COLS].sum()
    msg_df = df[df["usage_type"].isin(MSG_APPS)].groupby("customer_id")[ALL_OBS_COLS].sum()

    # Initialize Feature Matrix
    X = pd.DataFrame(index=data_df.index)

    # 4. Feature Generation Logic (From Notebook 1.0)
    logger.info("Calculating Weekly Aggregates...")
    X["data_w1"] = data_df[W1_COLS].sum(axis=1)
    X["data_w2"] = data_df[W2_COLS].sum(axis=1)
    X["data_w3"] = data_df[W3_COLS].sum(axis=1)
    X["data_w4"] = data_df[W4_COLS].sum(axis=1)

    X["spend_w1"] = spend_df[W1_COLS].sum(axis=1)
    X["spend_w2"] = spend_df[W2_COLS].sum(axis=1)
    X["spend_w3"] = spend_df[W3_COLS].sum(axis=1)
    X["spend_w4"] = spend_df[W4_COLS].sum(axis=1)

    logger.info("Calculating Trend Deltas...")
    # Data Trends
    X["trend_data_w2_vs_w1"] = safe_div((X["data_w2"] - X["data_w1"]), X["data_w1"])
    X["trend_data_w3_vs_w1"] = safe_div((X["data_w3"] - X["data_w1"]), X["data_w1"])
    X["trend_data_w4_vs_w1"] = safe_div(
        (X["data_w4"] - X["data_w1"]), X["data_w1"]
    )  # Key Predictor
    X["trend_data_w3_vs_w2"] = safe_div((X["data_w3"] - X["data_w2"]), X["data_w2"])
    X["trend_data_w4_vs_w2"] = safe_div((X["data_w4"] - X["data_w2"]), X["data_w2"])
    X["trend_data_w4_vs_w3"] = safe_div((X["data_w4"] - X["data_w3"]), X["data_w3"])

    # Spend Trends
    X["trend_spend_w2_vs_w1"] = safe_div((X["spend_w2"] - X["spend_w1"]), X["spend_w1"])
    X["trend_spend_w3_vs_w1"] = safe_div((X["spend_w3"] - X["spend_w1"]), X["spend_w1"])
    X["trend_spend_w4_vs_w1"] = safe_div((X["spend_w4"] - X["spend_w1"]), X["spend_w1"])
    X["trend_spend_w3_vs_w2"] = safe_div((X["spend_w3"] - X["spend_w2"]), X["spend_w2"])
    X["trend_spend_w4_vs_w2"] = safe_div((X["spend_w4"] - X["spend_w2"]), X["spend_w2"])
    X["trend_spend_w4_vs_w3"] = safe_div((X["spend_w4"] - X["spend_w3"]), X["spend_w3"])

    logger.info("Calculating Advanced Behavioral Features (Gini, Cliff, Volatility)...")

    # Gini Coefficient (Inequality)
    X["data_gini_coefficient"] = vectorized_gini(data_df)

    # Peak & Valley Weeks
    data_weeks = X[["data_w1", "data_w2", "data_w3", "data_w4"]]
    spend_weeks = X[["spend_w1", "spend_w2", "spend_w3", "spend_w4"]]

    X["peak_data_week"] = data_weeks.idxmax(axis=1).str.replace("data_w", "").astype("uint8")
    X["lowest_data_week"] = data_weeks.idxmin(axis=1).str.replace("data_w", "").astype("uint8")
    X["peak_spend_week"] = spend_weeks.idxmax(axis=1).str.replace("spend_w", "").astype("uint8")
    X["lowest_spend_week"] = spend_weeks.idxmin(axis=1).str.replace("spend_w", "").astype("uint8")

    # Cliff (Distance from Peak)
    peak_val_data = data_weeks.max(axis=1)
    peak_val_spend = spend_weeks.max(axis=1)
    X["drop_from_peak_data"] = safe_div((X["data_w4"] - peak_val_data), peak_val_data)
    X["drop_from_peak_spend"] = safe_div((X["spend_w4"] - peak_val_spend), peak_val_spend)

    # Volatility Shift
    X["data_volatility_shift"] = safe_div(
        data_df[W4_COLS].std(axis=1), data_df[W1_COLS].std(axis=1)
    )
    X["spend_volatility_shift"] = safe_div(
        spend_df[W4_COLS].std(axis=1), spend_df[W1_COLS].std(axis=1)
    )

    # Consistency Score
    X["data_consistency_score"] = safe_div(data_df.mean(axis=1), data_df.std(axis=1))
    X["spend_consistency_score"] = safe_div(spend_df.mean(axis=1), spend_df.std(axis=1))

    # Wallet Share (Video/Messaging Ratios)
    X["pct_video_w1"] = safe_div(video_df[W1_COLS].sum(axis=1), X["data_w1"])
    X["pct_video_w2"] = safe_div(video_df[W2_COLS].sum(axis=1), X["data_w2"])
    X["pct_video_w3"] = safe_div(video_df[W3_COLS].sum(axis=1), X["data_w3"])
    X["pct_video_w4"] = safe_div(video_df[W4_COLS].sum(axis=1), X["data_w4"])

    X["pct_messaging_w1"] = safe_div(msg_df[W1_COLS].sum(axis=1), X["data_w1"])
    X["pct_messaging_w2"] = safe_div(msg_df[W2_COLS].sum(axis=1), X["data_w2"])
    X["pct_messaging_w3"] = safe_div(msg_df[W3_COLS].sum(axis=1), X["data_w3"])
    X["pct_messaging_w4"] = safe_div(msg_df[W4_COLS].sum(axis=1), X["data_w4"])

    # Peak to Avg Ratios
    X["data_peak_to_avg_ratio"] = safe_div(data_df.max(axis=1), data_df.mean(axis=1))
    X["spend_peak_to_avg_ratio"] = safe_div(spend_df.max(axis=1), spend_df.mean(axis=1))

    # Min/Max Ratios
    avg_daily_data = data_df.mean(axis=1)
    min_daily_data = data_df[data_df > 0].min(axis=1).fillna(0)
    avg_daily_spend = spend_df.mean(axis=1)
    min_daily_spend = spend_df[spend_df > 0].min(axis=1).fillna(0)

    X["ratio_min_daily_data_to_avg"] = safe_div(min_daily_data, avg_daily_data)
    X["ratio_max_daily_data_to_avg"] = safe_div(data_df.max(axis=1), avg_daily_data)
    X["ratio_min_daily_spend_to_avg"] = safe_div(min_daily_spend, avg_daily_spend)
    X["ratio_max_daily_spend_to_avg"] = safe_div(spend_df.max(axis=1), avg_daily_spend)

    # Z-Scores (Last week vs History)
    HISTORY_COLS = W1_COLS + W2_COLS + W3_COLS

    X["data_w4_z_score"] = safe_div(
        (data_df[W4_COLS].mean(axis=1) - data_df[HISTORY_COLS].mean(axis=1)),
        data_df[HISTORY_COLS].std(axis=1),
    )
    X["spend_w4_z_score"] = safe_div(
        (spend_df[W4_COLS].mean(axis=1) - spend_df[HISTORY_COLS].mean(axis=1)),
        spend_df[HISTORY_COLS].std(axis=1),
    )

    # 5. Clean Up (Drop Raw Weekly Aggregates)
    # We drop raw volumes because we want unit-invariant features
    cols_to_drop = [
        "data_w1",
        "data_w2",
        "data_w3",
        "data_w4",
        "spend_w1",
        "spend_w2",
        "spend_w3",
        "spend_w4",
    ]
    X.drop(columns=cols_to_drop, inplace=True)
    X.fillna(0, inplace=True)

    # 6. Merge with Targets and Save
    logger.info("Merging with Targets and Saving...")

    train_targets = pd.read_parquet(PROCESSED_DATA_DIR / "train_targets.parquet")
    test_targets = pd.read_parquet(PROCESSED_DATA_DIR / "test_targets.parquet")

    # Align features with targets
    X_train = X.loc[train_targets["customer_id"]].reset_index()
    X_test = X.loc[test_targets["customer_id"]].reset_index()

    # Create final dataframes
    train_df = pd.merge(X_train, train_targets, on="customer_id")
    test_df = pd.merge(X_test, test_targets, on="customer_id")

    train_df.to_parquet(output_train_path)
    test_df.to_parquet(output_test_path)

    logger.success(
        f"Feature Engineering Complete. Train shape: {train_df.shape}, Test shape: {test_df.shape}"
    )


if __name__ == "__main__":
    app()

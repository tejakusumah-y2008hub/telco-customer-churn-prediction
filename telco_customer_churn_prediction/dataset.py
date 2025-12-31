from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from telco_customer_churn_prediction.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR,
    interim_output_path: Path = INTERIM_DATA_DIR / "combined_usage.parquet",
    process_targets: bool = True,
):
    """
    Loads raw data, merges usage files, and saves the combined interim dataset.
    """
    logger.info("--- Starting Data Ingestion ---")

    # 1. Define File Groups
    dataset_names = [
        "usage_app_facebook_daily",
        "usage_app_youtube_daily",
        "usage_app_tiktok_daily",
        "usage_app_whatsapp_daily",
        "usage_app_helakuru_daily",
        "usage_app_other",
        "usage_voice_d2d_incoming",
        "usage_voice_d2d_outgoing",
        "usage_voice_nd2d_incoming",
        "usage_voice_d2nd_outgoing",
        "usage_pack_data",
        "usage_pack_vas",
    ]

    # 2. Process Target Files (Train/Test Split IDs)
    if process_targets:
        logger.info("Processing Target Labels...")
        try:
            train_cxid = pd.read_csv(input_path / "train_cxid.csv")
            test_cxid = pd.read_csv(input_path / "test_cxid.csv")

            # Optimization: Downcast types
            train_cxid["churn"] = train_cxid["churn"].astype("int8")
            test_cxid["churn"] = test_cxid["churn"].astype("int8")

            # Save targets immediately to processed (they don't need feature eng)
            train_cxid.to_parquet(PROCESSED_DATA_DIR / "train_targets.parquet")
            test_cxid.to_parquet(PROCESSED_DATA_DIR / "test_targets.parquet")
            logger.success("Target labels saved to data/processed/")
        except FileNotFoundError:
            logger.error("Target files (train_cxid/test_cxid) not found!")
            raise

    # 3. Load and Merge Usage Data
    logger.info("Loading and Merging Usage DataFrames...")
    processed_dfs = []

    for name in dataset_names:
        file_p = input_path / f"{name}.csv"
        if not file_p.exists():
            logger.warning(f"File {name}.csv not found. Skipping.")
            continue

        logger.info(f"Loading {name}...")
        df = pd.read_csv(file_p)

        # Standardize column name
        if "customer" in df.columns:
            df.rename(columns={"customer": "customer_id"}, inplace=True)

        # Add a column to identify the usage type (needed for the long format)
        df["usage_type"] = name
        processed_dfs.append(df)

    if not processed_dfs:
        logger.error("No usage data loaded!")
        return

    # 4. Concatenate (Long Format)
    logger.info("Concatenating datasets...")
    combined_usage_df = pd.concat(processed_dfs, ignore_index=True)

    # Sort for efficiency
    combined_usage_df["usage_type"] = pd.Categorical(
        combined_usage_df["usage_type"], categories=dataset_names, ordered=True
    )
    combined_usage_df.sort_values(by=["customer_id", "usage_type"], inplace=True)

    # 5. Save Interim Data
    logger.info(f"Saving combined data to {interim_output_path}...")
    interim_output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_usage_df.to_parquet(interim_output_path)

    logger.success("Data Engineering Complete.")


if __name__ == "__main__":
    app()

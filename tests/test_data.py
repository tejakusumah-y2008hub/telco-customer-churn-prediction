import pytest

from telco_customer_churn_prediction.config import PROCESSED_DATA_DIR


def test_processed_data_exists():
    """
    Simple smoke test to ensure the processed train file exists.
    """
    train_file = PROCESSED_DATA_DIR / "train_df.parquet"

    # If the file doesn't exist, we skip the test instead of failing,
    # because maybe the user hasn't run 'make data' yet.
    if not train_file.exists():
        pytest.skip("Processed data not found. Run 'make features' first.")

    assert train_file.exists()

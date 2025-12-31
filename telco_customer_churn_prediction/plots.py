from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import typer

from telco_customer_churn_prediction.config import FIGURES_DIR, PROCESSED_DATA_DIR

try:
    import umap
except ImportError:
    umap = None

app = typer.Typer()


@app.command()
def plot_segmentation(
    input_path: Path = PROCESSED_DATA_DIR / "train_df.parquet",
    output_path: Path = FIGURES_DIR / "customer_segmentation_umap.png",
):
    """
    Generates the UMAP Customer Segmentation plot (Cluster 1 vs others).
    """
    if umap is None:
        logger.error("UMAP is not installed. Run `pip install umap-learn`.")
        return

    logger.info("Loading data for segmentation...")
    df = pd.read_parquet(input_path).fillna(0)

    # Use the features identified in Notebook 2.0
    features = [
        "trend_data_w4_vs_w1",
        "data_gini_coefficient",
        "peak_spend_week",
        "data_volatility_shift",
    ]

    X = df[features]
    y = df["churn"]

    logger.info("Running UMAP projection (this may take a moment)...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(StandardScaler().fit_transform(X))

    # Plotting
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="coolwarm", s=3, alpha=0.6)
    plt.title("Customer Segmentation Map (Risk Clusters)", fontsize=16)
    plt.colorbar(scatter, label="Churn Status (1=Churn)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    logger.success(f"Plot saved to {output_path}")


if __name__ == "__main__":
    app()

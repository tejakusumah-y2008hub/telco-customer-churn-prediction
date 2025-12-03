from pathlib import Path
import mlflow

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


def configure_mlflow():
    """
    Configures MLflow to save runs to the project root 'mlruns' folder.
    """
    # 1. Define the path using the existing PROJ_ROOT
    mlruns_dir = PROJ_ROOT / "mlruns"

    # 2. Convert to URI format (file:///E:/...)
    # Pathlib's .as_uri() handles the file:/// prefix and slashes automatically
    tracking_uri = mlruns_dir.as_uri()

    # 3. Set up MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Telco_Churn_Prediction_LKR")

    # 4. Use the existing logger instead of print()
    logger.info(f"MLflow configured. Tracking URI: {tracking_uri}")

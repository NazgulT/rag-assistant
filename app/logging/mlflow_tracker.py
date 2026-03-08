"""
MLFlow experiment tracking module.
"""
import os
from typing import Dict, Any, Optional
import mlflow
from app.logging.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)


class MLFlowTracker:
    """Tracker for MLFlow experiment tracking."""

    def __init__(
        self,
        experiment_name: str = settings.MLFLOW_EXPERIMENT_NAME,
        tracking_uri: str = settings.MLFLOW_TRACKING_URI,
    ):
        """
        Initialize MLFlow tracker.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: URI for MLFlow tracking
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        try:
            # Set tracking URI
            mlflow.set_tracking_uri(tracking_uri)

            # Set experiment
            mlflow.set_experiment(experiment_name)

            logger.info(
                f"MLFlowTracker initialized with experiment: {experiment_name}"
            )
        except Exception as e:
            logger.error(f"Error initializing MLFlowTracker: {str(e)}")
            raise

    def start_run(self, run_name: str = settings.MLFLOW_RUN_NAME):
        """Start a new MLFlow run."""
        try:
            mlflow.start_run(run_name=run_name)
            logger.info(f"Started MLFlow run: {run_name}")
        except Exception as e:
            logger.error(f"Error starting MLFlow run: {str(e)}")
            raise

    def end_run(self):
        """End the current MLFlow run."""
        try:
            mlflow.end_run()
            logger.info("Ended MLFlow run")
        except Exception as e:
            logger.error(f"Error ending MLFlow run: {str(e)}")
            raise

    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        try:
            mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Error logging parameters: {str(e)}")
            raise

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            logger.debug(f"Logged {len(metrics)} metrics")
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            raise

    def log_artifact(self, local_path: str):
        """Log artifact."""
        try:
            mlflow.log_artifact(local_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Error logging artifact: {str(e)}")
            raise

    def log_dict(self, data: Dict[str, Any], artifact_file: str):
        """Log dictionary as artifact."""
        try:
            mlflow.log_dict(data, artifact_file)
            logger.info(f"Logged dict to artifact: {artifact_file}")
        except Exception as e:
            logger.error(f"Error logging dict: {str(e)}")
            raise

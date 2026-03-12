"""
MLFlow experiment tracking module.
"""
import os
from typing import Dict, Any, Optional, ContextManager
import mlflow
from mlflow.tracking import MlflowClient
from app.logging.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)


class MLFlowTracker:
    """Tracker for MLFlow experiment tracking with trace support."""

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
        self.client = MlflowClient(tracking_uri)
        self.experiment_id = None

        try:
            # Set tracking URI
            mlflow.set_tracking_uri(tracking_uri)

            # Set experiment and get ID
            mlflow.set_experiment(experiment_name)
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
            
            logger.info(
                f"MLFlowTracker initialized with experiment: {experiment_name}"
            )
        except Exception as e:
            logger.error(f"Error initializing MLFlowTracker: {str(e)}")
            raise

    def set_experiment(self, experiment_name: str):
        """Set experiment and update experiment_id."""
        mlflow.set_experiment(experiment_name)
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment:
            self.experiment_id = experiment.experiment_id
        logger.info(f"Set experiment: {experiment_name}")

    def start_run(self, run_name: str = None, nested: bool = False) -> ContextManager:
        """
        Start a new MLFlow run.
        
        Can be used as context manager: with tracker.start_run():

        Args:
            run_name: Optional name for the run.
            nested: If True, start a nested run (requires a parent run to be active).
        """
        run_name = run_name or settings.MLFLOW_RUN_NAME
        try:
            # allow nested run option to avoid "already active" errors when
            # starting multiple runs inside a parent context.
            return mlflow.start_run(run_name=run_name, nested=nested)
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

    def get_run_id(self) -> str:
        """Get current run ID."""
        return mlflow.active_run().info.run_id if mlflow.active_run() else None

    def get_trace_url(self, run_id: str = None) -> str:
        """
        Get MLFlow trace URL for a run.
        
        Args:
            run_id: Run ID (uses current run if not provided)
            
        Returns:
            URL to view trace in MLFlow UI
        """
        if not run_id:
            run_id = self.get_run_id()
        
        if not run_id or not self.experiment_id:
            return None
        
        # MLFlow trace URL format
        tracking_uri = self.tracking_uri.rstrip('/')
        trace_url = f"{tracking_uri}/#/experiments/{self.experiment_id}/runs/{run_id}"
        return trace_url

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to the current run.

        If no run is active we simply log a warning and skip to avoid the
        MLFlow error that would start an implicit run.
        """
        if not mlflow.active_run():
            logger.warning("No active MLFlow run; parameters not logged")
            return
        try:
            mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Error logging parameters: {str(e)}")
            raise

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to the current run.

        Metrics are skipped if there is no active run. This prevents
        implicit runs from being started when callers forget to open a
        context.
        """
        if not mlflow.active_run():
            logger.warning("No active MLFlow run; metrics not logged")
            return
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

    def log_artifacts(self, artifacts: Dict[str, Any]):
        """
        Log multiple artifacts as dictionary.
        
        Args:
            artifacts: Dictionary of artifact names to values
        """
        try:
            for name, value in artifacts.items():
                if isinstance(value, dict):
                    mlflow.log_dict(value, f"{name}.json")
                else:
                    mlflow.log_text(str(value), f"{name}.txt")
            logger.debug(f"Logged {len(artifacts)} artifacts")
        except Exception as e:
            logger.error(f"Error logging artifacts: {str(e)}")
            raise

    def get_experiment_runs(self, limit: int = 100):
        """
        Get all runs in current experiment.
        
        Args:
            limit: Maximum number of runs to return
            
        Returns:
            List of run objects
        """
        if not self.experiment_id:
            return []
        
        try:
            return self.client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=["start_time DESC"],
                max_results=limit,
            )
        except Exception as e:
            logger.error(f"Error getting experiment runs: {str(e)}")
            return []

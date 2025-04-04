import os
import sys
import json
import mlflow
import mlflow.sklearn
from src.utils.logger import logger
from src.utils.exception import CustomException
import joblib 

class ModelRegister:
    def __init__(self, config):
        self.config = config

    def register(self):
        try:
            mlflow.set_tracking_uri(self.config.tracking['tracking_uri'])
            experiment_name = self.config.tracking.get("experiment_name", "Default")
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run():
                            if os.path.exists(self.config.model_dir):
                                model = joblib.load(self.config.model_dir)
                                logger.info("Loaded Model")
                            else:
                                error_msg = f"Model artifact not found at {self.config.model_dir}"
                                logger.info(error_msg)

                            mlflow.sklearn.log_model(model, "model")
                            mlflow.log_param("model_name", self.config.model_name)

                            # Load evaluation metrics from the artifacts folder if available
                            if os.path.exists(self.config.metrics_dir):
                                with open(self.config.metrics_dir, "r") as f:
                                    metrics = json.load(f)
                                for metric_name, metric_value in metrics.items():
                                    mlflow.log_metric(metric_name, float(metric_value))
                                mlflow.log_artifact(self.config.metrics_dir, artifact_path="metrics")
                                logger.info(f"Logged metrics from {self.config.metrics_dir}")  
                            else:
                              logger.info(f"Metrics file not found at {self.config.metrics_dir}") 
                            
                            run_id = mlflow.active_run().info.run_id
                            logger.info(f"Model logged to MLflow with run ID: {run_id}")

                            # Register model in MLflow Model Registry 
                            mlflow.register_model(f"runs:/{run_id}/model", self.config.model_name)
        except Exception as e:
            logger.error(f"Error during model registration: {e}")
            raise CustomException(e, sys)

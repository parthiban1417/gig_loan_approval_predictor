import os
import json
import sys
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys

class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def evaluate(self, X_test, y_test):
        try:
            logger.info("Evaluating model performance...")
            if os.path.exists(self.config.model_dir):
                            model = joblib.load(self.config.model_dir)
                            logger.info("Loaded Model")        
            else:
                 raise CustomException("Model not found at " + self.config.model_dir, sys)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = conf_matrix.ravel()
            logger.info(f"Model Accuracy: {accuracy:.4f}")
            
            # Create a dictionary of metrics
            metrics = {
                "accuracy": accuracy,
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "f1_score": report["weighted avg"]["f1-score"],
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp
            }

            def default_converter(o):
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.floating):
                    return float(o)
                raise TypeError(f"Object of type {type(o)} is not JSON serializable")
            
            # Define the metrics folder path and file path
            os.makedirs(self.config.metrics_dir, exist_ok=True)
            metrics_file_path = os.path.join(self.config.metrics_dir, self.config.metrics_name)
            
            # Save the metrics dictionary to a JSON file
            with open(metrics_file_path, "w") as f:
                json.dump(metrics, f, indent=4, default=default_converter)
            logger.info(f"Metrics stored at {metrics_file_path}")

        except Exception as e:
            logger.error("Error in model evaluation", exc_info=True)
            raise CustomException(e, sys)
        

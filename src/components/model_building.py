from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import logger
from src.utils.exception import CustomException
import os
import sys
import joblib

class ModelBuilding:
    def __init__(self, config):
        self.config = config
        self.model = None

    def build_model(self, X_train, y_train):
        try:
            logger.info("Building and training the model...")
            params = self.config.model_params
            self.model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                min_samples_leaf=params.get("min_samples_leaf", 1),
                max_features=params.get("max_features", "auto")
            )
            self.model.fit(X_train, y_train)
            logger.info("Model training complete.")

            # Save model to artifacts/model
            model_path = os.path.join(self.config.model_dir, self.config.model_name)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved at {model_path}")

        except Exception as e:
            logger.error("Error in model building")
            raise CustomException(e, sys)

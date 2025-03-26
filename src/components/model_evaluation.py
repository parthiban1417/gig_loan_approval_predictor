from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys

class ModelEvaluation:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):
        try:
            logger.info("Evaluating model performance...")
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            logger.info(f"Model Accuracy: {accuracy:.4f}")
            return {"accuracy": accuracy, "report": report, "confusion_matrix": conf_matrix}
        except Exception as e:
            logger.error("Error in model evaluation")
            raise CustomException(e, sys)

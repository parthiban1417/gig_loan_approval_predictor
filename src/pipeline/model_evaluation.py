from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import ConfigurationManager
from src.utils.logger import logger

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self, X_test, y_test):

        try:
            config_manager = ConfigurationManager()
            mb_config = config_manager.get_model_evaluation_config()
            evaluator = ModelEvaluation(mb_config)
            evaluator.evaluate(X_test, y_test)
            logger.info("Model Evaluation complete.")
            return 

        except Exception as e:
            logger.error("Model Evaluation failed.")
            raise e
        
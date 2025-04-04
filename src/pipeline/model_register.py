from src.components.model_register import ModelRegister
from src.config.configuration import ConfigurationManager
from src.utils.logger import logger

STAGE_NAME = "Model Register stage"

class ModelRegisterPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            mlflow_config = config_manager.get_mlflow_config()
            model_register = ModelRegister(mlflow_config)
            model_register.register()
            logger.info("Model Register complete.")

        except Exception as e:
            logger.error("Model Register failed.")
            raise e
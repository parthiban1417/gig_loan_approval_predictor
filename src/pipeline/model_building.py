from src.components.model_building import ModelBuilding
from src.config.configuration import ConfigurationManager
from src.utils.logger import logger

STAGE_NAME = "Model Building stage"

class ModelBuildingPipeline:
    def __init__(self):
        pass

    def main(self, X_train, y_train):
        try:
            config_manager = ConfigurationManager()
            mb_config = config_manager.get_model_building_config()
            model_builder = ModelBuilding(mb_config)
            model = model_builder.build_model(X_train, y_train)
            logger.info("Model Building complete.")

        except Exception as e:
            logger.error("Model Building failed.")
            raise e
        
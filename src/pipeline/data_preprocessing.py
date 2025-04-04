from src.components.data_preprocessing import DataPreprocessing
from src.config.configuration import ConfigurationManager
from src.utils.logger import logger

STAGE_NAME = "Data Preprocessing stage"

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            dp_config = config_manager.get_data_preprocessing_config()
            data_preprocessor = DataPreprocessing(dp_config)
            raw_data = dp_config.raw_data_file
            # Preprocess the full dataset using complete pipeline that returns train/test splits
            X_train, y_train, X_test, y_test = data_preprocessor.preprocess_pipeline(raw_data)
            logger.info("Data preprocessing complete.")
            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error("Data preprocessing failed.")
            raise e
        
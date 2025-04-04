from src.components.data_preprocessing import DataPreprocessing
from src.config.configuration import ConfigurationManager
from src.utils.logger import logger
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        pass

    def main(self, df: pd.DataFrame):
        try:
            config_manager = ConfigurationManager()
            dp_config = config_manager.get_data_preprocessing_config()
            data_preprocessor = DataPreprocessing(dp_config)
            processed_input = data_preprocessor.prediction_preprocess(df)
            logger.info("Prediction complete.")
            return processed_input
        
        except Exception as e:
            logger.error("Prediction failed.")
            raise e

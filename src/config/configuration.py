from src.utils.common import read_yaml
from src.entity.config_entity import (
    DataIngestionConfig, DataPreprocessingConfig, ModelBuildingConfig
)
from src.utils.logger import logger

CONFIG_FILE_PATH = "config/config.yaml"

class ConfigurationManager:
    def __init__(self):
        logger.info("Reading configuration file...")
        self.config = read_yaml(CONFIG_FILE_PATH)
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        di_config = self.config['data_ingestion']
        return DataIngestionConfig(
            root_dir=di_config['root_dir'],
            data_file=di_config['data_file']
        )
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        dp_config = self.config['data_preprocessing']
        return DataPreprocessingConfig(
            raw_data_file=dp_config['raw_data_file'],
            train_data_file=dp_config['train_data_file'],
            test_data_file=dp_config['test_data_file']
        )
    
    def get_model_building_config(self) -> ModelBuildingConfig:
        mb_config = self.config['model_building']
        return ModelBuildingConfig(model_params=mb_config['model_params'])
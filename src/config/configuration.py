from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (
    DataIngestionConfig, 
    DataPreprocessingConfig, 
    ModelBuildingConfig, 
    ModelEvaluationConfig, 
    DriftDetectionConfig,  
    MlflowConfig
)
from src.constants import CONFIG_FILE_PATH
from src.utils.logger import logger


class ConfigurationManager:
    def __init__(self):
        logger.info("Reading configuration file...")
        self.config = read_yaml(str(CONFIG_FILE_PATH))
        create_directories([self.config['artifacts_dir']])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        di_config = self.config['data_ingestion']
        return DataIngestionConfig(
            root_dir=di_config['root_dir'],
            data_file=di_config['data_file']
        )
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        dp_config = self.config['data_preprocessing']
        return DataPreprocessingConfig(
            root_dir=dp_config['root_dir'],
            raw_data_file=dp_config['raw_data_file'],
            train_data_file=dp_config['train_data_file'],
            test_data_file=dp_config['test_data_file'],
            transformer_name=dp_config['transformer_name']
        )
    
    def get_model_building_config(self) -> ModelBuildingConfig:
        mb_config = self.config['model_building']
        return ModelBuildingConfig(
            model_dir=mb_config['model_dir'],
            model_params=mb_config['model_params'],
            model_name=mb_config['model_name']
        )
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        me_config = self.config['model_evaluation']
        return ModelEvaluationConfig(
            model_dir=me_config['model_dir'],
            metrics_dir=me_config['metrics_dir'],
            metrics_name=me_config['metrics_name']
        )
    
    def get_model_drift_config(self) -> DriftDetectionConfig:
        md_config = self.config['drift_detection']
        return DriftDetectionConfig(
            drift_dir=md_config['drift_dir'],
            train_data_file=md_config['train_data_file'],
            test_data_file=md_config['test_data_file'],
            drift_name=md_config['drift_name']
        )
    
    def get_mlflow_config(self) -> MlflowConfig:
        ml_config = self.config['mlflow']
        return MlflowConfig( metrics_dir=ml_config['metrics_dir'],
                             model_name=ml_config['model_name'],
                             model_dir=ml_config['model_dir'],
                             tracking=ml_config['tracking']
        )
    
    def get_aws_config(self) -> dict:
        return self.config.get('aws', {}
        )
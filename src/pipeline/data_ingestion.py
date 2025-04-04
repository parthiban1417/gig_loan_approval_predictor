from src.components.data_ingestion import DataIngestion
from src.config.configuration import ConfigurationManager
from src.utils.logger import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info("Data Ingestion started.")
            config_manager = ConfigurationManager()
            di_config = config_manager.get_data_ingestion_config()
            data_ingestor = DataIngestion(di_config)
            data_ingestor.generate_synthetic_data()
            logger.info("Data Ingestion complete.")

        except Exception as e:
            logger.error("Data Ingestion failed.")
            raise e
        

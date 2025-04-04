from src.utils.logger import logger
from src.pipeline.data_ingestion import DataIngestionPipeline
from src.pipeline.data_preprocessing import DataPreprocessingPipeline
from src.pipeline.model_building import ModelBuildingPipeline
from src.pipeline.model_evaluation import ModelEvaluationPipeline
from src.pipeline.model_register import ModelRegisterPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f" {STAGE_NAME} started ") 
   data_ingestion = DataIngestionPipeline()
   data_ingestion.main()
   logger.info(f" {STAGE_NAME} completed ")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Data Preprocessing stage"
try:
   logger.info(f" {STAGE_NAME} started ")  
   data_preprocessing = DataPreprocessingPipeline()
   X_train, y_train, X_test, y_test = data_preprocessing.main()
   logger.info(f" {STAGE_NAME} completed ")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Model Building stage"
try:
   logger.info(f" {STAGE_NAME} started ")  
   model_building = ModelBuildingPipeline()
   model_building.main(X_train, y_train)
   logger.info(f" {STAGE_NAME} completed ")
except Exception as e:
        logger.exception(e)
        raise e




STAGE_NAME = "Model Evaluation stage"
try:
   logger.info(f" {STAGE_NAME} started ")  
   model_evaluation = ModelEvaluationPipeline()
   model_evaluation.main(X_test, y_test)
   logger.info(f" {STAGE_NAME} completed ")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Model Register stage"
try:
   logger.info(f" {STAGE_NAME} started ") 
   model_register = ModelRegisterPipeline()
   model_register.main()
   logger.info(f" {STAGE_NAME} completed ")
except Exception as e:
        logger.exception(e)
        raise e


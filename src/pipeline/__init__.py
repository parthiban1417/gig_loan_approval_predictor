# src/pipeline/__init__.py
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_building import ModelBuilding
from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import ConfigurationManager
from src.utils.logger import logger

def run_pipeline():
    try:
        logger.info("Pipeline started.")
        config_manager = ConfigurationManager()

        # Data Ingestion
        di_config = config_manager.get_data_ingestion_config()
        data_ingestor = DataIngestion(di_config)
        raw_data = data_ingestor.generate_synthetic_data()

        # ----- Data Preprocessing -----
        dp_config = config_manager.get_data_preprocessing_config()
        data_preprocessor = DataPreprocessing(dp_config)
        # Preprocess the full dataset using our complete pipeline that returns train/test splits
        X_train, y_train, X_test, y_test = data_preprocessor.preprocess_pipeline(raw_data, apply_smote=True)
        logger.info("Data preprocessing complete.")

        # Model Building
        mb_config = config_manager.get_model_building_config()
        model_builder = ModelBuilding(mb_config)
        model = model_builder.build_model(X_train, y_train)

        # Model Evaluation
        evaluator = ModelEvaluation(model)
        metrics = evaluator.evaluate(X_test, y_test)
        logger.info("Pipeline execution complete.")
        return model, metrics
    
    except Exception as e:
        logger.error("Pipeline execution failed.")
        raise e

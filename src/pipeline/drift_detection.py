from src.components.drift_detection import DriftDetection
from src.config.configuration import ConfigurationManager
from src.utils.logger import logger
import pandas as pd

STAGE_NAME = "Drift Detection stage"

class DriftDetectionPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Load drift detection configuration from config
            config_manager = ConfigurationManager()
            md_config = config_manager.get_model_drift_config()
            drift_detection = DriftDetection(md_config)
            
            # Read reference and current datasets
            ref_data = pd.read_csv(str(md_config.train_data_file))
            current_data = pd.read_csv(str(md_config.test_data_file))
            
            # Run drift detection
            drift_report = drift_detection.run_drift_detection(ref_data, current_data)
            logger.info("Drift detection complete.")
            return  drift_report
        
        except Exception as e:
            logger.error(f"Error during drift detection: {e}")
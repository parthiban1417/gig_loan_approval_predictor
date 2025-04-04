from src.components.drift_detection import DriftDetection
from src.config.configuration import ConfigurationManager
from src.utils.logger import logger

class DriftMetricsUpdater:
    def __init__(self):
        pass

    def main(self, registry, drift_gauge, drift_gauges: dict):
        try:
            # Load drift detection configuration from config
            config_manager = ConfigurationManager()
            md_config = config_manager.get_model_drift_config()
            drift_detect = DriftDetection(md_config)
            drift_metrics = drift_detect.update_drift_metrics(registry, drift_gauge, drift_gauges)
            logger.info("Drift metrics update complete.")
            return  drift_metrics
        
        except Exception as e:
            logger.error(f"Error during metrics: {e}")
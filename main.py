# main.py
from src.pipeline import run_pipeline
from src.utils.logger import logger

if __name__ == "__main__":
    try:
        logger.info("Main pipeline execution started.")
        model, metrics = run_pipeline()
        logger.info("Main pipeline execution finished successfully.")
        print("Model Metrics:")
        print(metrics)
    except Exception as e:
        logger.error("Main pipeline execution failed.")

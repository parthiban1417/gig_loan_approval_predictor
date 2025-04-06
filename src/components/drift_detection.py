import json
import os
import pandas as pd
import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from prometheus_client import Gauge
from src.utils.logger import logger
from src.utils.exception import CustomException

class DriftDetection:
    def __init__(self, config):
        self.config = config

    def run_drift_detection(self, ref_data: pd.DataFrame, current_data: pd.DataFrame) -> pd.DataFrame:
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_data, current_data=current_data)
            drift_report = report.as_dict()
            
            # Save the drift report to artifacts/drift/drift_report.json
            drift_report_path = os.path.join(self.config.drift_dir, self.config.drift_name)
            os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)
            with open(drift_report_path, "w") as f:
                json.dump(drift_report, f, indent=4)
            logger.info(f"Drift report saved at {drift_report_path}")
            return drift_report
        except Exception as e:
            logger.error("Error in drift detection: " + str(e))
            raise CustomException(e)
        
    def update_drift_metrics(self, registry, drift_gauge, drift_gauges: dict):
        """
        Update Prometheus drift gauges using values from drift_report.
        """
        drift_report_path = os.path.join(self.config.drift_dir, self.config.drift_name)
        if os.path.exists(drift_report_path):
            with open(drift_report_path, "r") as f:
                drift_report = json.load(f)
        
        # Process the list of metrics if it exists
        if "metrics" in drift_report:
            for item in drift_report["metrics"]:
                metric_name = item.get("metric", "").strip()
                result = item.get("result", {})
                # For the overall dataset drift metric
                if metric_name == "DatasetDriftMetric":
                    overall_score = result.get("drift_share", 0)
                    drift_gauge.labels(metric="overall").set(overall_score)
                # For the detailed drift table
                elif metric_name == "DataDriftTable":
                    # Update top-level details 
                    share = result.get("share_of_drifted_columns")
                    if share is not None:
                        if "share_of_drifted_columns" not in drift_gauges:
                            drift_gauges["share_of_drifted_columns"] = Gauge(
                                "share_of_drifted_columns",
                                "Share of drifted columns",
                                registry=registry
                            )
                        drift_gauges["share_of_drifted_columns"].set(share)
                    # Process per-column drift details
                    drift_by_columns = result.get("drift_by_columns", {})
                    for column, details in drift_by_columns.items():
                        # Create a gauge for the drift score of this column
                        gauge_name = f"drift_score_{column}"
                        if gauge_name not in drift_gauges:
                            drift_gauges[gauge_name] = Gauge(
                                gauge_name,
                                f"Drift score for {column}",
                                registry=registry
                            )
                        drift_gauges[gauge_name].set(details.get("drift_score", 0))
                        
        else:
            for metric, value in drift_report.items():
                if isinstance(value, dict):
                    for sub_metric, sub_value in value.items():
                        label = f"{metric}_{sub_metric}"
                        if label not in drift_gauges:
                            drift_gauges[label] = Gauge(
                                label,
                                f"Drift metric for {label}",
                                registry=registry
                            )
                        drift_gauges[label].set(sub_value)
                else:
                    if metric not in drift_gauges:
                        drift_gauges[metric] = Gauge(
                            metric,
                            f"Drift metric for {metric}",
                            registry=registry
                        )
                    drift_gauges[metric].set(value)

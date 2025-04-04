from flask import Flask, request, render_template, redirect, url_for ,Response
import numpy as np
import pandas as pd
from src.utils.logger import logger
from src.pipeline.drift_detection import DriftDetectionPipeline
from src.pipeline.drift_metrics import DriftMetricsUpdater
from src.pipeline.prediction import PredictionPipeline
import os
import joblib
import psutil
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry

app = Flask(__name__)


# Load model at startup
MODEL_PATH = os.path.join("artifacts", "model", "gigloanpredictormodel.pkl")
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully for serving.")
except Exception as e:
    logger.error("Error loading model:", e)

# Set up Prometheus registry and gauges
registry = CollectorRegistry()
drift_gauge = Gauge('evidently_data_drift_score', 'Data drift score from Evidently AI', ['metric'], registry=registry)
# Drift metrics: update these dynamically based on drift report.
# For each drift metric (e.g., overall, or per-feature), a Gauge is created.
drift_gauges = {}
# System usage metrics
cpu_usage_gauge = Gauge('system_cpu_usage_percent', 'System CPU usage percent', registry=registry)
memory_usage_gauge = Gauge('system_memory_usage_percent', 'System memory usage percent', registry=registry)

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/train',methods=['GET', 'POST'])  # route to train the pipeline
def training():
    os.system("python main.py")
    logger.info("Training completed. Redirecting to prediction page.")
    return redirect(url_for('predict'))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Handles prediction requests.
    """
    if request.method == "POST":
        # Assume the input form has fields for each feature
        try:
            # Retrieve input values from the form
            input_data = {
                            'age': int(request.form['age']),
                            'education_level': request.form['education_level'],
                            'num_platforms': float(request.form['num_platforms']),
                            'work_experience': float(request.form['work_experience']),
                            'monthly_income': float(request.form['monthly_income']),
                            'seasonal_variation': float(request.form['seasonal_variation']),
                            'income_volatility': int(request.form['income_volatility']),
                            'savings_balance': float(request.form['savings_balance']),
                            'debt_to_income_ratio': float(request.form['debt_to_income_ratio']),
                            'credit_score': int(request.form['credit_score']),
                            'existing_loans': float(request.form['existing_loans']),
                            'loan_amount_requested': float(request.form['loan_amount_requested']),
                            'transaction_frequency': int(request.form['transaction_frequency']),
                            'avg_monthly_expenses': float(request.form['avg_monthly_expenses']),
                            'credit_card_utilization': int(request.form['credit_card_utilization']),
                            'subscription_services': int(request.form['subscription_services']),
                            'financial_emergencies_last_year': int(request.form['financial_emergencies_last_year']),
                            'inflation_rate': float(request.form['inflation_rate']),
                            'customer_feedback_score': float(request.form['customer_feedback_score']),
                            'work_consistency': int(request.form['work_consistency']),
                            'penalties': float(request.form['penalties']),
                            'alternative_income_source': request.form['alternative_income_source'],
                            'loan_coapplicant': request.form['loan_coapplicant'],
                            'urban_rural': request.form['urban_rural'],
                            'avg_platform_tenure': float(request.form['avg_platform_tenure']),
                            'family_dependents': int(request.form['family_dependents']),
                            'cost_of_living_index': float(request.form['cost_of_living_index']),
                            'reason_for_loan': request.form['reason_for_loan'],
                            'platform_ratings': request.form['platform_ratings']
                        }
            # Convert input data into DataFrame for prediction
            input_df = pd.DataFrame([input_data])
            logger.info(f"Input DataFrame for prediction:\n{input_df}")
            # Preprocess input data using the loaded preprocessor
            predictor = PredictionPipeline()
            processed_input = predictor.main(input_df)
            logger.info(f"Processed input data: {processed_input}")
            prediction = model.predict(processed_input)
            logger.info(f"Predicted output: {prediction[0]}")
            result = "Approved" if prediction[0] == 1 else "Rejected"
            return render_template("result.html", result=result)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return render_template("result.html", result="Error in prediction.")
    return render_template("predict.html")

@app.route("/drift", methods=["GET"])
def drift():
        try:
            # Load drift detection configuration from config
            logger.info(f" Drift Detection started ") 
            drift_detection = DriftDetectionPipeline()
            drift_report = drift_detection.main()
            logger.info(f" Drift Detection completed ")
        
            return render_template("drift.html", drift_report=drift_report)
    
        except Exception as e:
            logger.error(f"Error during drift detection: {e}")
            return render_template("drift.html", drift_report={"error": str(e)})
        
@app.route("/metrics")
def metrics():
    """
    Exposes system and drift metrics so that Prometheus (and Grafana) can scrape them.
    """
    try:
        # Update system metrics with current values
        cpu_usage = psutil.cpu_percent(interval=1)
        mem_usage = psutil.virtual_memory().percent
        cpu_usage_gauge.set(cpu_usage)
        memory_usage_gauge.set(mem_usage)
        # Update drift metrics from a saved report file if it exists
        metrics_updater = DriftMetricsUpdater()
        metrics_updater.main(registry, drift_gauge, drift_gauges)
    except Exception as e:
        logger.exception("Error updating metrics: %s", e)

    return Response(generate_latest(registry), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 1417, debug=True, use_reloader=False)




from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.components.model_building import ModelBuilding
from src.components.data_preprocessing import DataPreprocessing
from src.config.configuration import ConfigurationManager
from src.utils.logger import logger

app = Flask(__name__)

# For this local demo, we'll run the pipeline on startup
config_manager = ConfigurationManager()

def load_model():
    """
    Runs the pipeline to build and return a trained model.
    For production, persist the model and load it instead.
    """
    from src.pipeline import run_pipeline
    return run_pipeline()[0]

dp_config = config_manager.get_data_preprocessing_config()
data_preprocessor = DataPreprocessing(dp_config)

# Load the model on app start
model = load_model()
logger.info("Model loaded for serving in the local app.")

@app.route("/", methods=["GET", "POST"])
def index():
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
            processed_input = data_preprocessor.prediction_preprocess(input_df)
            logger.info(f"Processed input data: {processed_input}")
          
            prediction = model.predict(processed_input)
            result = "Approved" if prediction[0] == 1 else "Rejected"
            return render_template("result.html", result=result)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return render_template("result.html", result="Error in prediction.")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 1417, debug=True)




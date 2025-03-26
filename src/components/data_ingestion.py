import os
import numpy as np
import pandas as pd
import random
from scipy.stats import poisson
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys

class DataIngestion:
    def __init__(self, config):
        """
        Initialize DataIngestion with the provided configuration.
        """
        self.config = config
    
    def generate_synthetic_data(self):
        try:
            logger.info("Starting synthetic data generation...")
            # Set seed for reproducibility
            np.random.seed(42)
            random.seed(42)

            # Number of samples
            n = 5000

            # ---------------------------
            # 1. Basic Applicant Information
            # ---------------------------
            applicant_id = np.arange(1, n + 1)
            age = np.random.randint(22, 60, n)
            education_level = np.random.choice(
                ["High School", "Graduate", "Postgraduate"],
                n,
                p=[0.4, 0.45, 0.15]
            )

            # ---------------------------
            # 2. Gig Work Details
            # ---------------------------
            # List of popular gig platforms in India
            all_platforms = [
                "Swiggy", "Zomato", "Rapido", "Ola", "Uber",
                "Amazon Flex", "Dunzo", "UrbanClap", "Fiverr", "Upwork"
            ]

            def pick_platforms():
                count = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
                return random.sample(all_platforms, count)

            gig_platforms_list = [pick_platforms() for _ in range(n)]
            gig_platforms = [", ".join(platforms) for platforms in gig_platforms_list]
            num_platforms = [len(platforms) for platforms in gig_platforms_list]
            work_experience = np.random.randint(0, 15, n)  # in years

            # ---------------------------
            # 3. Financial Information with Seasonal Effects
            # ---------------------------
            # Base monthly income using a normal distribution (in INR)
            base_income = np.random.normal(loc=30000, scale=10000, size=n).astype(int)
            base_income = np.clip(base_income, 5000, 100000)

            # Seasonal variation factor: simulate peaks (range from 0.8 to 1.5)
            seasonal_variation = np.random.choice(np.linspace(0.8, 1.5, 15), n)

            # Adjust monthly income for multi-platform engagement (10% boost per extra platform)
            monthly_income = (base_income * seasonal_variation * (1 + 0.1 * (np.array(num_platforms) - 1))).astype(int)

            # Income volatility: proportional to income (5% to 30% variability)
            income_volatility = (monthly_income * np.random.uniform(0.05, 0.3, n)).astype(int)

            # Savings balance: roughly 6 months of income times a factor between 0.2 and 0.5
            savings_balance = (monthly_income * 6 * np.random.uniform(0.2, 0.5, n)).astype(int)
            savings_balance = np.clip(savings_balance, 0, 500000)

            # Simulate existing loans realistically
            existing_loans = np.random.choice([0, 1, 2, 3], size=n, p=[0.7, 0.2, 0.08, 0.02])

            # Calculate Debt-to-Income Ratio (DTI)
            dti_base = np.random.uniform(0.1, 0.3, n)
            debt_to_income_ratio = np.clip(dti_base * (1 + 0.2 * existing_loans), 0.1, 0.6)

            # Loan amount requested (in INR) with microfinance focus: 70% micro (below 50k), 30% regular
            micro_loans = np.random.randint(10000, 50000, int(n * 0.7))
            regular_loans = np.random.randint(50000, 500000, n - int(n * 0.7))
            loan_amount_requested = np.concatenate([micro_loans, regular_loans])
            np.random.shuffle(loan_amount_requested)

            # ---------------------------
            # 4. Credit Score Calculation (CIBIL-like)
            # ---------------------------
            credit_score = (
                300 +
                (monthly_income / 100) -
                (debt_to_income_ratio * 50) +
                np.random.normal(0, 30, n)
            )
            credit_score = np.clip(credit_score, 300, 900).astype(int)

            # ---------------------------
            # 5. Behavioral and Economic Factors
            # ---------------------------
            transaction_frequency = np.random.randint(10, 100, n)
            avg_monthly_expenses = np.random.randint(5000, 70000, n)
            credit_card_utilization = np.random.randint(10, 90, n)
            subscription_services = np.random.randint(0, 5, n)
            financial_emergencies_last_year = np.random.randint(0, 5, n)
            # Enhanced inflation rates (capped at 7.8% per RBI data)
            inflation_rate = np.round(np.random.uniform(3, 7.8, n), 2)

            # ---------------------------
            # 6. Additional Untraditional Parameters
            # ---------------------------
            loan_reasons = [
                "Vehicle Purchase", "Medical Emergency", "Education", 
                "Home Renovation", "Debt Consolidation", "Business Expansion", "Other"
            ]
            reason_for_loan = np.random.choice(loan_reasons, n)

            def generate_platform_ratings(platforms):
                ratings = {p: round(np.random.uniform(3.0, 5.0), 1) for p in platforms}
                return "; ".join([f"{p}:{r}" for p, r in ratings.items()])

            platform_ratings = [generate_platform_ratings(platforms) for platforms in gig_platforms_list]
            customer_feedback_score = [
                round(np.mean([float(r.split(":")[1]) for r in ratings.split("; ")]) * 20 + np.random.uniform(-5, 5), 1)
                for ratings in platform_ratings
            ]
            work_consistency = np.random.randint(1, 8, n)
            penalties = [max(0, int(np.random.poisson(1) - (score / 100))) for score in customer_feedback_score]
            alternative_income_source = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
            loan_coapplicant = np.random.choice(["Yes", "No"], n, p=[0.2, 0.8])

            # ---------------------------
            # 7. Enhanced Geographic and Additional Features
            # ---------------------------
            indian_states = [
                'Maharashtra', 'Karnataka', 'Delhi', 'Tamil Nadu', 'Uttar Pradesh',
                'Gujarat', 'West Bengal', 'Telangana', 'Rajasthan', 'Bihar'
            ]
            location = np.random.choice(indian_states, n, p=[0.18, 0.15, 0.12, 0.1, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05])
            urban_ratio = 0.35
            urban_rural = np.random.choice(['Urban', 'Rural'], n, p=[urban_ratio, 1 - urban_ratio])

            def generate_platform_tenure(platforms):
                return [random.randint(3, 60) for _ in platforms]

            platform_tenures = [generate_platform_tenure(platforms) for platforms in gig_platforms_list]
            avg_platform_tenure = [np.mean(tenures) for tenures in platform_tenures]

            family_dependents = poisson.rvs(mu=1.5, size=n)
            family_dependents = np.clip(family_dependents, 0, 5)

            cost_of_living_index = {
                'Maharashtra': 1.15, 'Karnataka': 1.1, 'Delhi': 1.25,
                'Tamil Nadu': 1.05, 'Uttar Pradesh': 0.95, 'Gujarat': 1.0,
                'West Bengal': 0.9, 'Telangana': 1.07, 'Rajasthan': 0.93, 'Bihar': 0.85
            }
            cost_of_living = np.array([cost_of_living_index[state] for state in location])

            # ---------------------------
            # 8. Loan Approval Outcome with Logical Checks
            # ---------------------------
            def determine_loan_approval(i):
                # Basic financial criteria
                crit_credit = credit_score[i] > 650
                crit_dti = debt_to_income_ratio[i] < 0.4
                crit_savings = savings_balance[i] > (loan_amount_requested[i] * 0.2)
                crit_feedback = customer_feedback_score[i] > 70
                crit_consistency = work_consistency[i] >= 3

                # Logical checks:
                crit_min_credit = credit_score[i] >= 500  # Very low credit scores should be rejected

                # If no co-applicant, require a higher credit score
                if loan_coapplicant[i] == "No":
                    crit_coapplicant = credit_score[i] > 700
                else:
                    crit_coapplicant = True

                # Combine all criteria
                if all([crit_credit, crit_dti, crit_savings, crit_feedback, crit_consistency, 
                        crit_min_credit, crit_coapplicant]):
                    return 1
                else:
                    return 0

            loan_approved = [determine_loan_approval(i) for i in range(n)]

            # ---------------------------
            # 9. Create Final DataFrame and Introduce Missing Values
            # ---------------------------
            df = pd.DataFrame({
                "applicant_id": applicant_id,
                "age": age,
                "education_level": education_level,
                "gig_platforms": gig_platforms,
                "num_platforms": num_platforms,
                "work_experience": work_experience,
                "monthly_income": monthly_income,
                "seasonal_variation": seasonal_variation,
                "income_volatility": income_volatility,
                "savings_balance": savings_balance,
                "debt_to_income_ratio": debt_to_income_ratio,
                "credit_score": credit_score,
                "existing_loans": existing_loans,
                "loan_amount_requested": loan_amount_requested,
                "transaction_frequency": transaction_frequency,
                "avg_monthly_expenses": avg_monthly_expenses,
                "credit_card_utilization": credit_card_utilization,
                "subscription_services": subscription_services,
                "financial_emergencies_last_year": financial_emergencies_last_year,
                "inflation_rate": inflation_rate,
                "reason_for_loan": reason_for_loan,
                "platform_ratings": platform_ratings,
                "customer_feedback_score": customer_feedback_score,
                "work_consistency": work_consistency,
                "penalties": penalties,
                "alternative_income_source": alternative_income_source,
                "loan_coapplicant": loan_coapplicant,
                "location": location,
                "urban_rural": urban_rural,
                "avg_platform_tenure": avg_platform_tenure,
                "family_dependents": family_dependents,
                "cost_of_living_index": cost_of_living,
                "loan_approved": loan_approved
            })

            # ---------------------------
            # 10. Introduce Missing Values (Enhanced Pattern)
            # ---------------------------
            missing_config = {
                'education_level': 0.08,
                'work_experience': 0.08,
                'monthly_income': 0.12,
                'savings_balance': 0.12,
                'credit_score': 0.12,
                'avg_monthly_expenses': 0.12,
                'urban_rural': 0.05,
                'family_dependents': 0.03
            }

            for col, ratio in missing_config.items():
                df.loc[df.sample(frac=ratio).index, col] = np.nan

            # ---------------------------
            # 11. Validation Checks (Sanity Checks)
            # ---------------------------
            # 1. Inflation rate sanity check (should not exceed 7.8%)
            assert df.inflation_rate.max() <= 7.8, "Inflation rate exceeds RBI cap."

            # 2. Micro-loans proportion: ~70% loans should be under ₹50k
            micro_proportion = (df.loan_amount_requested < 50000).mean()
            assert 0.65 <= micro_proportion <= 0.75, "Micro-loan proportion out of range."

            # 3. Regional cost of living: sample check for Delhi (if any records exist)
            if (df.location == 'Delhi').any():
                delhi_cost = df[df.location == 'Delhi'].cost_of_living_index.mean()
                assert 1.2 < delhi_cost < 1.3, "Delhi cost of living anomaly."

            # ---------------------------
            # 12. Save Final Dataset
            # ---------------------------
            os.makedirs(os.path.dirname(self.config.data_file), exist_ok=True)
            df.to_csv(self.config.data_file, index=False)
            logger.info(f"Synthetic data generated and saved to {self.config.data_file}")
            return df
        
        except Exception as e:
            logger.error("Error in Data Ingestion")
            raise CustomException(e, sys)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from src.utils.logger import logger
from src.utils.exception import CustomException
import os
import sys
import joblib 

class DataPreprocessing:
    def __init__(self, config):
        """
        Initialize with Data Preprocessing configuration.
        :param config: Configuration object with attributes:
            - raw_data_file: path to raw synthetic data
            - train_data_file: path to save training data
            - test_data_file: path to save testing data
        """
        self.config = config

    def split_data(self, df: pd.DataFrame):
        try:
            logger.info("Splitting data into train and test sets...")
            logger.info("Data Preprocessing config type: %s", type(self.config))
            logger.info("Data Preprocessing config value: %s", self.config)
            train, test = train_test_split(df, test_size=0.2, stratify=df['loan_approved'], random_state=42)
            os.makedirs(os.path.dirname(self.config.train_data_file), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.test_data_file), exist_ok=True)
            train.to_csv(self.config.train_data_file, index=False)
            test.to_csv(self.config.test_data_file, index=False)
            logger.info("Train-test split complete and saved.")
            return train, test
        except Exception as e:
            logger.error("Error in train-test splitting")
            raise CustomException(e, sys)

    def _base_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input DataFrame.
        :param df: Input DataFrame (either train or test subset)
        :param apply_smote: If True, apply SMOTE oversampling (recommended only on training data)
        :return: Tuple of (features X, target y)
        """
        try:
            logger.info("Starting data preprocessing on a subset...")


            # === Step 3: One-Hot Encoding for 'reason_for_loan' ===
            df = pd.get_dummies(df, columns=['reason_for_loan'], drop_first=True)

            # === Step 4: Fraud Flag & Missing Value Handling for Fraud Cases ===
            # If work_experience==0 and monthly_income is missing then force loan_approved=0
            df.loc[(df['work_experience'] == 0) & (df['monthly_income'].isnull()), 'loan_approved'] = 0
            df['fraud_flag'] = ((df['work_experience'] == 0) & (df['monthly_income'].isnull())).astype(int)
            # Split fraud vs non-fraud
            non_fraud_data = df[df['fraud_flag'] == 0].copy()
            fraud_data = df[df['fraud_flag'] == 1].copy()
            # For fraud cases, fill missing monthly_income with 0
            fraud_data['monthly_income'] = fraud_data['monthly_income'].fillna(0)
            # For non-fraud, use median for monthly_income and work_experience
            non_fraud_data['monthly_income'].fillna(non_fraud_data['monthly_income'].median(), inplace=True)
            non_fraud_data['work_experience'].fillna(non_fraud_data['work_experience'].median(), inplace=True)
            # Also, if work_experience == 0, impute using median
            non_fraud_data.loc[non_fraud_data['work_experience'] == 0, 'work_experience'] = non_fraud_data['work_experience'].median()
            # Recombine the data
            df = pd.concat([non_fraud_data, fraud_data], ignore_index=True)

            # === Step 5: First-Time Applicants Flag & Credit Score Adjustment ===
            df['first_time_applicant'] = ((df['work_experience'] == 0) & (df['existing_loans'] == 0)).astype(int)
            df.loc[df['first_time_applicant'] == 1, 'credit_score'] = -1

            # === Step 6: Further Missing Value Imputation ===
            df['savings_balance'].fillna(df['savings_balance'].median(), inplace=True)
            df['avg_monthly_expenses'].fillna(df['avg_monthly_expenses'].median(), inplace=True)
            df['credit_score'].fillna(df['credit_score'].median(), inplace=True)
            df['urban_rural'].fillna(df['urban_rural'].mode()[0], inplace=True)
            df['family_dependents'].fillna(df['family_dependents'].mode()[0], inplace=True)
            df['education_level'].fillna(df['education_level'].mode()[0], inplace=True)

            # === Step 7: Process Platform Ratings ===
            # Split the 'platform_ratings' string into key-value pairs
            platform_ratings = df['platform_ratings'].str.split('; ', expand=True)
            platform_ratings_dict = platform_ratings.applymap(lambda x: dict([x.split(':')]) if pd.notna(x) else {})
            # Function to calculate average rating from a dictionary
            def calculate_avg(ratings):
                if isinstance(ratings, dict) and len(ratings) > 0:
                    return sum(map(float, ratings.values())) / len(ratings)
                return None
            platform_ratings_dict['avg_platform_rating'] = platform_ratings_dict[[0,1,2]].applymap(calculate_avg).mean(axis=1)
            platform_ratings_dict.drop([0,1,2], axis=1, inplace=True)
            df = pd.concat([df, platform_ratings_dict], axis=1)

            # === Step 8: Drop Unnecessary Columns ===
            df.drop(columns=['first_time_applicant', 'fraud_flag', 'applicant_id', 'gig_platforms', 'platform_ratings', 'location'], inplace=True)

            # === Step 9: Encode Categorical Variables 
            df['education_level'] = df['education_level'].map({'High School': 0, 'Graduate': 1, 'Postgraduate': 2})
            df['urban_rural'] = df['urban_rural'].map({'Urban': 1, 'Rural': 0})
            df['alternative_income_source'] = df['alternative_income_source'].map({'Yes': 1, 'No': 0})
            df['loan_coapplicant'] = df['loan_coapplicant'].map({'Yes': 1, 'No': 0})
            df['education_level'] = df['education_level'].astype('int')
            df['alternative_income_source'] = df['alternative_income_source'].astype('int')
            df['loan_coapplicant'] = df['loan_coapplicant'].astype('int')
            df['urban_rural'] = df['urban_rural'].astype('int')
            df['family_dependents'] = df['family_dependents'].astype('int')

            # === Step 11: Log-Transform Skewed Numerical Features ===
            df['savings_balance'] = np.log1p(df['savings_balance'])
            df['existing_loans'] = np.log1p(df['existing_loans'])
            df['loan_amount_requested'] = np.log1p(df['loan_amount_requested'])
            df['penalties'] = np.log1p(df['penalties'])
            df['num_platforms'] = np.log1p(df['num_platforms'])
            df['loan_coapplicant'] = np.log1p(df['loan_coapplicant'])

            # === Step 12: Split Features and Target ===
            X = df.drop('loan_approved', axis=1)
            y = df['loan_approved']

            logger.info("Data preprocessing complete for this subset.")
            return X, y

        except Exception as e:
            logger.error("Error in data preprocessing")
            raise CustomException(e, sys)
        
    def preprocess_train(self, df: pd.DataFrame, apply_smote: bool = True):
        """
        Preprocess training data.
        Fits the PowerTransformer on the training set.
        :param df: Training DataFrame.
        :param apply_smote: Whether to apply SMOTE oversampling (recommended only on training data).
        :return: Tuple (X_train, y_train, fitted PowerTransformer)
        """
        logger.info("Preprocessing training data...")
        X, y = self._base_preprocessing(df)
        # Fit PowerTransformer on 'credit_score'
        pt = PowerTransformer(method='yeo-johnson')
        X['credit_score'] = pt.fit_transform(X[['credit_score']])
        transformer_path = os.path.join(self.config.root_dir, self.config.transformer_name)
        os.makedirs(os.path.dirname(transformer_path), exist_ok=True)
        joblib.dump(pt, transformer_path)
        logger.info("Saved PowerTransformer")
        if apply_smote:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
        logger.info("Training data preprocessing complete.")
        return X, y, pt

    def preprocess_test(self, df: pd.DataFrame, pt: PowerTransformer):
        """
        Preprocess test data using a fitted PowerTransformer.
        :param df: Test DataFrame.
        :param pt: Fitted PowerTransformer from the training set.
        :return: Tuple (X_test, y_test)
        """
        logger.info("Preprocessing test data...")
        X, y = self._base_preprocessing(df)
        X['credit_score'] = pt.transform(X[['credit_score']])
        logger.info("Test data preprocessing complete.")
        return X, y

    def preprocess_pipeline(self, df: pd.DataFrame, apply_smote: bool = True):
        """
        Complete preprocessing pipeline: split data, preprocess train and test sets.
        :param df: Full dataset.
        :param apply_smote: Whether to apply SMOTE on training data.
        :return: Tuple (X_train, y_train, X_test, y_test)
        """
        logger.info("Running complete preprocessing pipeline...")
        df = pd.read_csv(df)
        train_df, test_df = self.split_data(df)
        X_train, y_train, pt = self.preprocess_train(train_df, apply_smote)
        X_test, y_test = self.preprocess_test(test_df, pt)
        logger.info("Preprocessing pipeline complete.")
        return X_train, y_train, X_test, y_test
  
    def prediction_preprocess(self, df: pd.DataFrame):
                    try:
                        logger.info("Running prediction preprocessing pipeline...")
                        REASONS_FOR_LOAN = ['Debt Consolidation', 'Education', 'Home Renovation', 'Medical Emergency', 'Other', 'Vehicle Purchase']
              
                        # === Step 2: One-Hot Encoding for 'Reason for Loan' ===
                        selected_reason = df['reason_for_loan'].iloc[0]
                        for reason in REASONS_FOR_LOAN:
                            df[f'reason_for_loan_{reason}'] = 1 if reason == selected_reason else 0


                        # === Step 3: Handle Platform Ratings ===
                        def calculate_average_platform_rating(rating_str: str) -> float:
                            """
                            Parses a platform ratings string and returns the average rating.
                            Example input: "Zomato:4.7; Swiggy:3.3"
                            """
                            try:
                                # Split the string by semicolon and strip whitespace
                                items = [item.strip() for item in rating_str.split(';') if item.strip()]
                                ratings = []
                                for item in items:
                                    # Split each item into platform and score
                                    parts = item.split(':')
                                    if len(parts) == 2:
                                        try:
                                            ratings.append(float(parts[1].strip()))
                                        except ValueError:
                                            continue
                                # Return the average rating if ratings exist, otherwise 0
                                return np.mean(ratings) if ratings else 0.0
                            except Exception as e:
                                # In case of any error, log and return 0.0
                                logger.info(f"Error parsing platform ratings: {e}")
                                return 0.0
                            
                        # === Step 4: Encode Categorical Variables ===
                        df['education_level'] = df['education_level'].map({'High School': 0, 'Graduate': 1, 'Postgraduate': 2}).astype(int)
                        df['urban_rural'] = df['urban_rural'].map({'Urban': 1, 'Rural': 0}).astype(int)
                        df['alternative_income_source'] = df['alternative_income_source'].map({'Yes': 1, 'No': 0}).astype(int)
                        df['loan_coapplicant'] = df['loan_coapplicant'].map({'Yes': 1, 'No': 0}).astype(int)

                        # === Step 5: Log-Transform Skewed Numerical Features ===
                        for col in ['savings_balance', 'existing_loans', 'loan_amount_requested', 'penalties', 'num_platforms']:
                            if col in df.columns:
                                df[col] = np.log1p(df[col])
                        
                        transformer_path = os.path.join(self.config.root_dir, self.config.transformer_name)
                        if os.path.exists(transformer_path):
                            pt = joblib.load(transformer_path)
                            logger.info("Loaded PowerTransformer")
                            df['credit_score'] = pt.transform(df[['credit_score']])
                        else:
                            raise CustomException("PowerTransformer not found at " + transformer_path, sys)
                        
                        df['avg_platform_rating'] = df['platform_ratings'].apply(calculate_average_platform_rating)

                        # Drop column 
                        df.drop(columns=['reason_for_loan','platform_ratings'], inplace=True)
                        logger.info("Preprocessing Preprocessing pipeline complete.")
                        return df
                    
                    except Exception as e:
                        logger.error("Error in data preprocessing")
                        raise CustomException(e, sys)




































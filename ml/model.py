# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import logging

from db.database_operations import (
    get_database_session,
    update_or_create_last_trained_row_id,
    get_current_last_trained_row_id,
)
from db.setup import History

logger = logging.getLogger(__name__)


class MainML:
    """
    Main class that orchestrates model training, evaluation, and prediction.
    """

    def __init__(self, df, model_path):
        self.last_trained_row_id = get_current_last_trained_row_id()
        self.df = df
        self.model_path = model_path
        self.xgb_model = XGBClassifier(random_state=42)
        logger.info("MainML instance created.")

    def train_and_save_model(self, features, target):
        """
        Trains the XGBoost model and saves it to the specified path.
        """
        logger.info("Starting model training and saving process.")

        # Split the dataset into features (X) and target (y)
        X = self.df[features]
        y = self.df[target]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info("Data split into training and testing sets.")

        # Train the model
        self.xgb_model.fit(X_train, y_train)
        logger.info("Model training completed.")

        # Save the model
        joblib.dump(self.xgb_model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

        # Evaluate the model on the test set
        self.evaluate_model(X_test, y_test)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model on the test data and prints the classification report and confusion matrix.
        """
        logger.info("Starting model evaluation.")

        # Make predictions on the test set
        y_pred = self.xgb_model.predict(X_test)

        # Log classification report and confusion matrix
        logger.info(
            "XGBoost Classification Report:\n%s", classification_report(y_test, y_pred)
        )
        logger.info("XGBoost Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

    def load_model(self):
        """
        Loads the model from the specified path.
        """
        self.xgb_model = joblib.load(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")

    def predict(self, new_data):
        """
        Predicts the class for the new data point.
        """
        logger.info("Generating predictions for new data.")

        # Ensure that the new_data has the same features as the training set
        prediction = self.xgb_model.predict(new_data)
        probability = self.xgb_model.predict_proba(new_data)
        logger.info("Predictions generated: %s", prediction)
        logger.info("Prediction probabilities generated: %s", probability)
        return prediction, probability

    def incremental_train_with_new_data(self, batch_size=50):
        """
        Incrementally updates the XGBoost model with new data when `batch_size` rows are available.
        """
        logger.info("Checking for new data to update model incrementally.")

        session = get_database_session()

        # Start a new session
        try:
            # Fetch new data rows since the last trained row
            new_data = (
                session.query(History)
                .filter(
                    History.actual_result.is_not(None),
                    History.id > self.last_trained_row_id,
                )
                .order_by(History.id.asc())
                .limit(batch_size)
                .all()
            )

            # Proceed only if there are enough new rows
            if len(new_data) == batch_size:
                logger.info(
                    f"Found {batch_size} new rows. Starting incremental training."
                )

                new_data = [record.__dict__ for record in new_data]

                # Remove SQLAlchemy metadata (e.g., _sa_instance_state)
                for record in new_data:
                    record.pop("_sa_instance_state", None)

                # Convert new data to DataFrame
                new_df = pd.DataFrame(
                    new_data,
                    columns=[
                        "radiant_avg_hero_winrate",
                        "radiant_avg_roshans_killed",
                        "radiant_avg_last_hits",
                        "radiant_avg_denies",
                        "radiant_avg_hero_damage",
                        "radiant_avg_gpm",
                        "radiant_avg_xpm",
                        "radiant_avg_net_worth",
                        "radiant_avg_player_level",
                        "radiant_sum_obs",
                        "radiant_sum_sen",
                        "radiant_avg_teamfight_participation_cols",
                        "dire_avg_hero_winrate",
                        "dire_avg_roshans_killed",
                        "dire_avg_last_hits",
                        "dire_avg_denies",
                        "dire_avg_hero_damage",
                        "dire_avg_gpm",
                        "dire_avg_xpm",
                        "dire_avg_net_worth",
                        "dire_avg_player_level",
                        "dire_sum_obs",
                        "dire_sum_sen",
                        "dire_avg_teamfight_participation_cols",
                        "radiant_avg_kda",
                        "dire_avg_kda",
                        "actual_result",
                    ],
                )
                new_df = new_df.rename(columns={"actual_result": "radiant_win"})

                features = new_df.columns.drop("radiant_win").tolist()
                target = "radiant_win"
                X_new = new_df[features]
                y_new = new_df[target]

                new_xgb_model = XGBClassifier(random_state=42)
                loaded_xgb_model = joblib.load(self.model_path)

                new_xgb_model.fit(X_new, y_new, xgb_model=loaded_xgb_model)
                # Incrementally train the model
                self.xgb_model = new_xgb_model
                logger.info("Incremental model training completed.")

                # Save the updated model
                joblib.dump(self.xgb_model, self.model_path)
                logger.info(f"Incrementally updated model saved to {self.model_path}")

                # Update last trained row ID
                self.last_trained_row_id = new_data[-1]["id"]
                update_or_create_last_trained_row_id(self.last_trained_row_id)
                logger.info(
                    f"Model updated with new data up to row_id {self.last_trained_row_id}"
                )
            else:
                logger.info(
                    "Not enough new data for incremental training. Waiting for more rows."
                )
        finally:
            session.close()

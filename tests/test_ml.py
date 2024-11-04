# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

import unittest
from dataclasses import dataclass
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from ml.model import MainML, logger


class TestMainML(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataset for testing
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        self.df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        self.df["target"] = y
        self.model_path = "dummy_model_path.pkl"
        self.main_ml = MainML(self.df, self.model_path)
        self.main_ml.xgb_model = XGBClassifier()
        self.main_ml.last_trained_row_id = 0

        X_dummy = pd.DataFrame([[0], [1]], columns=["awg"])
        y_dummy = [0, 1]
        self.main_ml.xgb_model.fit(X_dummy, y_dummy)
        self.main_ml.xgb_model.save_model(self.model_path)

    @patch.object(XGBClassifier, "predict", return_value=np.array([0, 1] * 10))
    @patch("joblib.dump")
    @patch.object(XGBClassifier, "fit")
    def test_train_and_save_model(self, mock_fit, mock_joblib_dump, mock_predict):
        # logger = logging.getLogger('ml.model')
        with self.assertLogs(logger, level="INFO") as log:
            self.main_ml.train_and_save_model(
                features=[f"feature_{i}" for i in range(5)], target="target"
            )

            # Check logs for model save and evaluation
            self.assertIn(f"INFO:ml.model:Model saved to {self.model_path}", log.output)

        # Ensure fit and joblib.dump are called
        mock_fit.assert_called_once()
        mock_joblib_dump.assert_called_once_with(
            self.main_ml.xgb_model, self.model_path
        )

    @patch("joblib.load")
    def test_load_model(self, mock_joblib_load):
        mock_joblib_load.return_value = XGBClassifier()
        with self.assertLogs(logger, level="INFO") as log:
            self.main_ml.load_model()
            self.assertIn(
                f"INFO:ml.model:Model loaded from {self.model_path}", log.output
            )

        mock_joblib_load.assert_called_once_with(self.model_path)

    @patch.object(XGBClassifier, "predict", return_value=np.array([1]))
    @patch.object(XGBClassifier, "predict_proba", return_value=np.array([[0.2, 0.8]]))
    def test_predict(self, mock_predict_proba, mock_predict):
        new_data = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

        predictions, probabilities = self.main_ml.predict(new_data)

        # Assert the predict method was called correctly
        mock_predict.assert_called_once_with(new_data)
        mock_predict_proba.assert_called_once_with(new_data)

        # Assert predictions and probabilities are as expected
        self.assertEqual(predictions[0], 1)
        np.testing.assert_array_equal(probabilities, [[0.2, 0.8]])

    @patch.object(XGBClassifier, "predict", return_value=np.array([0, 1]))
    def test_evaluate_model(self, mock_predict):
        X_test = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]])
        y_test = np.array([0, 1])
        with self.assertLogs(logger, level="INFO") as log:
            self.main_ml.evaluate_model(X_test, y_test)

            # Check for parts of the expected log output instead of exact matches
            classification_report_logged = any(
                "XGBoost Classification Report:" in message for message in log.output
            )
            confusion_matrix_logged = any(
                "XGBoost Confusion Matrix:" in message for message in log.output
            )

            self.assertTrue(
                classification_report_logged, "Classification report not found in logs"
            )
            self.assertTrue(
                confusion_matrix_logged, "Confusion matrix not found in logs"
            )

    @patch.object(
        XGBClassifier, "fit"
    )  # Mock the `fit` method to avoid actual training
    @patch("ml.model.get_database_session")  # Mock the database session creation
    @patch("joblib.load")  # Mock `joblib.load` to avoid loading from file
    def test_incremental_training_with_enough_data(
        self, mock_joblib_load, mock_get_session, mock_fit
    ):
        # Create a mock session and mock return for `get_database_session`
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Create a mock model to be returned by `joblib.load`
        mock_model = XGBClassifier()
        mock_joblib_load.return_value = mock_model

        # Define mock data similar to your database records
        @dataclass
        class HistoryRecord:
            id: int
            match_id: int
            actual_result: int
            awg: float

        mock_data = [
            HistoryRecord(id=1, match_id=1, actual_result=1, awg=0.1),
            HistoryRecord(id=2, match_id=2, actual_result=0, awg=0.1),
            HistoryRecord(id=3, match_id=3, actual_result=1, awg=0.1),
            HistoryRecord(id=4, match_id=4, actual_result=0, awg=0.1),
            HistoryRecord(id=5, match_id=5, actual_result=1, awg=0.1),
        ]

        # Setup the session mock to return our mock history data
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = (
            mock_data
        )

        # Call the method under test
        with self.assertLogs("ml.model", level="INFO") as log:
            self.main_ml.incremental_train_with_new_data(batch_size=5)

            # Check that the model `fit` method was called once
            mock_fit.assert_called_once()

            # Optionally, verify that specific log messages were emitted, if needed
            assert any(
                "Found 5 new rows. Starting incremental training." in message
                for message in log.output
            )

    @patch("joblib.dump")
    @patch.object(XGBClassifier, "fit")
    @patch("ml.model.get_database_session")
    def test_incremental_training_with_no_data(
        self, mock_get_session, mock_fit, mock_joblib_dump
    ):
        # Mock the session and return a mock session object
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Mock no new data
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = (
            []
        )

        with self.assertLogs(logger, level="INFO") as log:
            self.main_ml.incremental_train_with_new_data(batch_size=5)

            # Ensure fit is not called
            mock_fit.assert_not_called()
            self.assertIn(
                "INFO:ml.model:Not enough new data for incremental training. Waiting for more rows.",
                log.output,
            )

import unittest
from unittest.mock import patch
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
            self.assertIn(
                "INFO:ml.model:Model saved to dummy_model_path.pkl", log.output
            )

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
                "INFO:ml.model:Model loaded from dummy_model_path.pkl", log.output
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

import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from ml.model import MainML  # assuming your class is saved in a file named main_ml.py


class TestMainML(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataset for testing
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        self.df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        self.df["target"] = y
        self.model_path = "dummy_model_path.pkl"
        self.main_ml = MainML(self.df, self.model_path)

    @patch.object(XGBClassifier, "predict")  # Mock predict method
    @patch("joblib.dump")
    @patch.object(XGBClassifier, "fit")
    @patch("builtins.print")  # Mock print to capture outputs
    def test_train_and_save_model(
        self, mock_print, mock_fit, mock_joblib_dump, mock_predict
    ):
        # Mock predict to return the correct number of predictions
        mock_predict.return_value = np.array(
            [0] * 10 + [1] * 10
        )  # 20 predictions to match y_test size

        # Test the training and saving of the model
        self.main_ml.train_and_save_model(
            features=[f"feature_{i}" for i in range(5)], target="target"
        )

        # Ensure the fit method was called during training
        mock_fit.assert_called_once()

        # Ensure the model was saved to the specified path
        mock_joblib_dump.assert_called_once_with(
            self.main_ml.xgb_model, self.model_path
        )

        # Ensure evaluate_model was called (check print statements for classification report/confusion matrix)
        mock_print.assert_any_call("Model saved to dummy_model_path.pkl")

    @patch("joblib.load")
    @patch("builtins.print")  # Mock print to capture outputs
    def test_load_model(self, mock_print, mock_joblib_load):
        # Mock the joblib.load to return a dummy model
        mock_joblib_load.return_value = XGBClassifier()

        # Call the method to load the model
        self.main_ml.load_model()

        # Ensure the model was loaded
        mock_joblib_load.assert_called_once_with(self.model_path)

        # Ensure print statement was called
        mock_print.assert_any_call("Model loaded from dummy_model_path.pkl")

    @patch.object(XGBClassifier, "predict", return_value=np.array([1]))
    def test_predict(self, mock_predict):
        # Create dummy new data for prediction
        new_data = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

        # Call the predict method
        prediction = self.main_ml.predict(new_data)

        # Ensure predict method was called
        mock_predict.assert_called_once_with(new_data)

        # Check if the prediction result is as expected
        self.assertEqual(prediction[0], 1)

    @patch.object(XGBClassifier, "predict", return_value=np.array([0, 1]))
    @patch("builtins.print")  # Mock print to capture outputs
    def test_evaluate_model(self, mock_print, mock_predict):
        # Create test data for evaluation
        X_test = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]])
        y_test = np.array([0, 1])

        # Call the evaluate_model method
        self.main_ml.evaluate_model(X_test, y_test)

        # Ensure predict method was called
        mock_predict.assert_called_once_with(X_test)

        # Check if classification report and confusion matrix were printed
        mock_print.assert_any_call("XGBoost Classification Report:")
        mock_print.assert_any_call("XGBoost Confusion Matrix:")

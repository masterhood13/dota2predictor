# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib


class MainML:
    """
    Main class that orchestrates model training, evaluation, and prediction.
    """

    def __init__(self, df, model_path):
        self.df = df
        self.model_path = model_path
        self.xgb_model = XGBClassifier(random_state=42)

    def train_and_save_model(self, features, target):
        """
        Trains the XGBoost model and saves it to the specified path.
        """
        # Split the dataset into features (X) and target (y)
        X = self.df[features]
        y = self.df[target]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        self.xgb_model.fit(X_train, y_train)

        # Save the model
        joblib.dump(self.xgb_model, self.model_path)
        print(f"Model saved to {self.model_path}")

        # Evaluate the model on the test set
        self.evaluate_model(X_test, y_test)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model on the test data and prints the classification report and confusion matrix.
        """
        # Make predictions on the test set
        y_pred = self.xgb_model.predict(X_test)

        # Print classification report
        print("XGBoost Classification Report:")
        print(classification_report(y_test, y_pred))

        # Print confusion matrix
        print("XGBoost Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def load_model(self):
        """
        Loads the model from the specified path.
        """
        self.xgb_model = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")

    def predict(self, new_data):
        """
        Predicts the class for the new data point.
        """
        # Ensure that the new_data has the same features as the training set
        prediction = self.xgb_model.predict(new_data)
        return prediction

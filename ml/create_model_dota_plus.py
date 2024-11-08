# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

import os
import logging
import pandas as pd
from ml.model import MainML
from structure.helpers import prepare_match_prediction_data, remove_zero_columns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

file_path = os.path.join("..", "dataset", "train_data", "all_data_match_predict.csv")
scaler_path = "../scaler_dota_plus.pkl"

# Load and prepare the dataset
df = pd.read_csv(file_path)

df = prepare_match_prediction_data(df, scaler_path)

df = remove_zero_columns(df)

# features = df.columns[:-1].tolist()  # All columns except the last one
target = "radiant_win"  # Change to your target column name
features = df.columns.drop(target).tolist()

# Path to save the model
model_path = "../xgb_model_dota_plus.pkl"  # Path where the model will be saved

# # Create an instance of MainML
main_ml = MainML(df, model_path)

# Train and save the model
main_ml.train_and_save_model(features, target)

# Load the model
main_ml.load_model()

# Prepare new data for prediction (replace this with actual data)
new_data = df.tail(5).drop(
    columns=[target]
)  # Assuming the last row is new data to predict
prediction = main_ml.predict(new_data)

print(f"Prediction for new data: {prediction}")
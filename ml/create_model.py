# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

import os
import pandas as pd
from ml.model import MainML
from structure.helpers import prepare_data


file_path = os.path.join("..", "dataset", "train_data", "all_data.csv")
scaler_path = "../scaler.pkl"

# Load and prepare the dataset
df = pd.read_csv(file_path)
df = prepare_data(df, scaler_path)

# Specify the features and target column
# features = df.columns[:-1].tolist()  # All columns except the last one
target = "radiant_win"  # Change to your target column name
features = df.columns.drop(target).tolist()

# Path to save the model
model_path = "../xgb_model.pkl"  # Path where the model will be saved

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

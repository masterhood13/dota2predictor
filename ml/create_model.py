# © 2024 Viktor Hamretskyi masterhood13@gmail.com
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.



import os
import pandas as pd

from ml.model import MainML
from structure.helpers import prepare_data

# Define file paths
file_path = os.path.join("..", "dataset", "train_data", "all_data.csv")

# Load and prepare the dataset
df = pd.read_csv(file_path)
df = prepare_data(df)

# Define hyperparameters and paths
model_path = "../my_pytorch_model.pth"
seed = 17
epochs = 60
weight_decay = 1e-5
batch_size = 10
dropout = 0.2
hidden_layers = [100, 50]

# Create an instance of MainML with the required arguments
predictor = MainML(
    df=df,  # The DataFrame after preparation
    model_path=model_path,  # Path where the model will be saved
    top_features=None,  # Use all features if None is passed
    n_hidden=hidden_layers,  # Hidden layers configuration
    drop_p=dropout,  # Dropout rate
    random_state=seed,  # Seed for reproducibility
)

# Train and save the model
predictor.train_and_save_model(epochs=epochs, batch_size=batch_size, learning_rate=1e-5)


predictor = MainML(
    df=df,  # The DataFrame after preparation
    model_path=model_path,  # Path where the model will be saved
    top_features=None,  # Use all features if None is passed
    n_hidden=hidden_layers,  # Hidden layers configuration
    drop_p=dropout,  # Dropout rate
    random_state=seed,  # Seed for reproducibility
)
# Load the saved model and evaluate it on the test set
predictor.load_model_and_evaluate()

predictor = MainML(
    df=df.tail(1),  # The DataFrame after preparation
    model_path=model_path,  # Path where the model will be saved
    top_features=None,  # Use all features if None is passed
    n_hidden=hidden_layers,  # Hidden layers configuration
    drop_p=dropout,  # Dropout rate
    random_state=seed,  # Seed for reproducibility
)
predictions = predictor.predict_new_data(df.tail(1))
print(f"Predictions: {predictions}")

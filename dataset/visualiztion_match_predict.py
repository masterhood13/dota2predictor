# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from structure.helpers import prepare_match_prediction_data


pd.set_option("display.max_columns", None)

file_path = os.path.join("..", "dataset", "train_data", "all_data.csv")
scaler_path = "../scaler.pkl"

# Load and prepare the dataset
df = pd.read_csv(file_path)
df = prepare_match_prediction_data(df, scaler_path)

# Specify the features and target column
# features = df.columns[:-1].tolist()  # All columns except the last one
target = "radiant_win"  # Change to your target column name
features = df.columns.drop(target).tolist()


corr_matrix = df.corr()

# Visualize with a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix[["radiant_win"]].sort_values(by="radiant_win", ascending=False),
    annot=True,
    cmap="coolwarm",
)
plt.title("Correlation of Features with Radiant Win")
plt.show()

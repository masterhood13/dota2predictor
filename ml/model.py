from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
import random
import torch
import numpy as np
from torch.utils import data
from torch import optim


def set_random_seed(rand_seed=17):
    """Helper function for setting random seed. Use for reproducibility of results"""
    if type(rand_seed) == int:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)


class MLP(nn.Module):
    """Multi-layer perceptron with ReLu and Softmax.

    Parameters:
    -----------
        n_input (int): number of nodes in the input layer
        n_hidden (int list): list of number of nodes n_hidden[i] in the i-th hidden layer
        n_output (int):  number of nodes in the output layer
        drop_p (float): drop-out probability [0, 1]
        random_state (int): seed for random number generator (use for reproducibility of result)
    """

    def __init__(self, n_input, n_hidden, drop_p, random_state=17):
        super().__init__()
        self.random_state = random_state
        set_random_seed(17)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_input, n_hidden[0])])
        self.hidden_layers.extend(
            [nn.Linear(h1, h2) for h1, h2 in zip(n_hidden[:-1], n_hidden[1:])]
        )
        self.output_layer = nn.Linear(n_hidden[-1], 1)
        self.dropout = nn.Dropout(p=drop_p)  # method to prevent overfitting

    def forward(self, X):
        """Forward propagation -- computes output from input X."""
        for h in self.hidden_layers:
            X = F.relu(h(X))
            X = self.dropout(X)
        X = self.output_layer(X)
        return torch.sigmoid(X)

    def predict_proba(self, X_test):
        return self.forward(X_test).detach().squeeze(1).cpu().numpy()


class Dota2MatchPredictor:
    """
    Multi-layer perceptron with ReLU and Softmax.

    Parameters:
    -----------
        df (pd.DataFrame): The DataFrame containing the features and target variable.
        top_features (list): List of feature names to be used for prediction.
        n_hidden (int list): List of number of nodes in each hidden layer.
        drop_p (float): Drop-out probability [0, 1].
        random_state (int): Seed for random number generator (use for reproducibility of results).
    """

    def __init__(
        self,
        df,
        top_features=None,
        n_hidden=[0, 60, 60],
        drop_p=0.001,
        random_state=42,
    ):
        self.df = df
        self.top_features = (
            top_features
            if top_features
            else df.columns[df.columns != "radiant_win"].tolist()
        )
        self.n_hidden = n_hidden
        self.drop_p = drop_p
        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    def get_x_and_y(self):
        # Feature matrix (X) and target vector (y)
        X = self.df[self.top_features].values
        try:
            y = self.df["radiant_win"].values.astype(np.float32)
        except KeyError:
            y = None
        return X, y

    def create_data_loaders(self, X, y, batch_size=10, test_size=0.1):
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # Create DataLoader objects
        train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = data.TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        return train_loader, test_loader

    def train(self, model_path, epochs=100, batch_size=10, learning_rate=0.00001):
        X, y = self.get_x_and_y()
        train_loader, test_loader = self.create_data_loaders(
            X, y, batch_size=batch_size
        )

        # Define the model
        input_size = X.shape[1]
        model = MLP(n_input=input_size, n_hidden=self.n_hidden, drop_p=self.drop_p)

        # Set up the loss function and optimizer
        criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze(1)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(
                f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}"
            )

        # Save the model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Evaluate the model
        self.evaluate(model, test_loader)

    def evaluate(self, model, test_loader):
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch).squeeze(1)
                predictions = (outputs > 0.5).float()
                y_pred.extend(predictions.numpy())
                y_true.extend(y_batch.numpy())

        accuracy = np.mean(np.array(y_pred) == np.array(y_true))
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

    def load_model(self, model_path, input_size):
        model = MLP(n_input=input_size, n_hidden=self.n_hidden, drop_p=self.drop_p)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Model loaded from {model_path}")
        return model

    def predict(self, model, new_data):
        new_data = new_data[self.top_features]
        model.eval()
        new_data_tensor = torch.tensor(new_data.to_numpy(), dtype=torch.float32)
        with torch.no_grad():
            predictions = model(new_data_tensor).squeeze(1)
            return predictions.numpy()


class MainML:
    """
    Main class that orchestrates model training, evaluation, and prediction.
    """

    def __init__(
        self,
        df,
        model_path,
        top_features=None,
        n_hidden=[100, 50],
        drop_p=0.2,
        random_state=17,
    ):
        self.df = df
        self.model_path = model_path
        self.dota_predictor = Dota2MatchPredictor(
            df,
            top_features=top_features,
            n_hidden=n_hidden,
            drop_p=drop_p,
            random_state=random_state,
        )

    def train_and_save_model(self, epochs=100, batch_size=10, learning_rate=0.00001):
        self.dota_predictor.train(
            self.model_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

    def load_model_and_evaluate(self):
        X, _ = self.dota_predictor.get_x_and_y()
        input_size = X.shape[1]
        model = self.dota_predictor.load_model(self.model_path, input_size=input_size)
        _, test_loader = self.dota_predictor.create_data_loaders(X, _)
        self.dota_predictor.evaluate(model, test_loader)

    def predict_new_data(self, new_data):
        X, _ = self.dota_predictor.get_x_and_y()
        input_size = X.shape[1]
        model = self.dota_predictor.load_model(self.model_path, input_size=input_size)
        predictions = self.dota_predictor.predict(model, new_data)
        return predictions

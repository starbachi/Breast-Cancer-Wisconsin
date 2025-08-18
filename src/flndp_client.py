import numpy as np
import pandas as pd
import flwr as fl
import torch.nn as nn

import random, torch

from typing import Dict, Tuple, List
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from flwr.common import NDArrays, Scalar
from flwr.client import NumPyClient
from flndp_model import BreastCancerMLP

class FLClient(NumPyClient):
    def __init__(self, client_id: str, X_train: np.ndarray, y_train: np.ndarray, device: str = "cpu", RANDOM_SEED: int = 1):
        """Initialise the client with training data and model.
        Args:
            client_id (str): Unique identifier for the client.
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            device (str): Device to run the model on ("cpu" or "cuda").
            RANDOM_SEED (int): Seed for reproducibility.
        """
        # Initialise client attributes
        self.client_id = client_id
        self.device = torch.device(device)
        self.seed = RANDOM_SEED + hash(client_id) % 1000
        
        # Set client-specific seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # --------------------------- PREPARE TRAINING DATA -------------------------- #
        # Convert training data to numpy arrays if they are not already
        self.X_train = X_train.to_numpy() if isinstance(X_train, (pd.DataFrame, pd.Series)) else np.array(X_train)
        self.y_train = y_train.to_numpy() if isinstance(y_train, (pd.DataFrame, pd.Series)) else np.array(y_train)
        
        # Ensure training data is 2D
        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        # ---------------------------------------------------------------------------- #
        
        # Print simple data statistics
        malignant_count = int((y_train == 0).sum())
        benign_count = int((y_train == 1).sum())
        print(f"Client {client_id}: {len(X_train)} samples, {benign_count} benign, {malignant_count} malignant")
            
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return client's model parameters used for training as numpy arrays for server aggregation."""
        try:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        except Exception as e:
            print(f"Error getting parameters for {self.client_id}: {str(e)}")
            raise
    
    def set_parameters(self, parameters: NDArrays) -> None:
        if not hasattr(self, "model"):
            # Lazy init: infer architecture from server parameter shapes
            arch_config = {"input_dim": parameters[0].shape[0], "hidden_dims": [10, 5]}
            self.model = BreastCancerMLP(arch_config=arch_config).to(self.device)
        
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v, device=self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def _create_local_model(self, arch_config: dict) -> nn.Module:
        """Create a local model instance based on the architecture configuration."""
        try:
            model = BreastCancerMLP(arch_config=arch_config)
            model.to(self.device)
            print(f"Local model created for client {self.client_id} with architecture: {arch_config}")
            return model
        except Exception as e:
            print(f"Error creating local model for {self.client_id}: {str(e)}")
            raise

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on local data and return updated parameters, number of samples, and metrics.
        Args:
            parameters (NDArrays): Model parameters from the server.
            config (Dict[str, Scalar]): Configuration parameters for training.
        Returns:
            Tuple[NDArrays, int, Dict[str, Scalar]]: Updated model parameters, number of samples used for training, and metrics.
        """
        try:
            self.set_parameters(parameters)
            
            # -------------------------- TRAINING CONFIGURATION -------------------------- #
            # Ensure required keys are present in config
            required_keys = ["local_epochs", "batch_size", "learning_rate", "optimizer", "loss"]
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required config key: {key} for client {self.client_id}")
                
            # Assign training parameters from config
            epochs = int(config["local_epochs"])
            batch_size = int(config["batch_size"])
            learning_rate = float(config["learning_rate"])

            # Get loss function class from config
            loss_fn_name = config.get("loss")
            loss_cls = getattr(torch.nn, str(loss_fn_name))
            self.criterion = loss_cls()

            # Get optimizer class from config
            optimizer_name = config.get("optimizer", "Adam")
            optimizer_cls = getattr(torch.optim, str(optimizer_name))
            optimizer = optimizer_cls(self.model.parameters(), lr=learning_rate)
            
            # Convert training data to tensors
            self.X_train = torch.tensor(self.X_train, dtype = torch.float32).to(self.device)
            self.y_train = torch.tensor(self.y_train, dtype = torch.float32).unsqueeze(1).to(self.device)

            # Create data loader
            dataset = TensorDataset(self.X_train, self.y_train)
            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True, generator=torch.Generator().manual_seed(self.seed))
            
            # Training loop
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            for _ in range(epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()   # Reset gradients
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                total_loss += epoch_loss
            
            avg_loss = total_loss / max(num_batches, 1)
            
            # Return updated parameters, number of samples, and metrics
            return (
                self.get_parameters({}),
                len(self.X_train),
                {"loss": avg_loss}
            )
            
        except Exception as e:
            # Log the error and raise it
            print(f"Training failed for {self.client_id}: {str(e)}")
            raise
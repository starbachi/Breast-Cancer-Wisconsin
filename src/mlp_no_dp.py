import torch
import torch.nn as nn
import numpy as np
import flwr as fl
from typing import List, Optional, Union


class BreastCancerMLP(nn.Module):
    """
    Multi-Layer Perceptron for breast cancer classification.
    
    This neural network is designed for binary classification of breast cancer
    samples using the Wisconsin Breast Cancer dataset features. The architecture
    uses two hidden layers with ReLU activation and dropout regularization.
    
    Architecture: Input(30) -> Hidden1(64) -> Hidden2(32) -> Output(1)
    
    Args:
        input_dim (int): Number of input features (default: 30 for breast cancer dataset)
        hidden1 (int): Number of neurons in first hidden layer (default: 64)
        hidden2 (int): Number of neurons in second hidden layer (default: 32)
        dropout_rate (float): Dropout probability for regularization (default: 0.3)
    """
    
    def __init__(self, input_dim: int = 30, hidden1: int = 64, hidden2: int = 32, dropout_rate: float = 0.3):
        super(BreastCancerMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout_rate = dropout_rate
        
        # Sequential network architecture
        self.network = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Hidden Layer 2 
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output Layer with BCEWithLogitsLoss
            nn.Linear(hidden2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 1)
        """
        return self.network(x)
    
    def get_parameter_count(self) -> tuple[int, int]:
        """
        Get the total and trainable parameter counts for the model.
        
        Returns:
            tuple[int, int]: (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def print_architecture(self) -> None:
        """Print a summary of the model architecture."""
        total_params, trainable_params = self.get_parameter_count()
        
        print(f" Input features: {self.input_dim}")
        print(f" Hidden Layer 1: {self.hidden1} neurons")
        print(f" Hidden Layer 2: {self.hidden2} neurons") 
        print(f" Output Layer: 1 neuron")
        print(f" Activation: ReLU + Dropout({self.dropout_rate})")
        print(f" Total parameters: {total_params:,}")
        print(f" Trainable parameters: {trainable_params:,}")


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extract model parameters as list of numpy arrays for Flower.
    
    This function extracts the model's parameters and converts them to numpy
    arrays, which is the format expected by the Flower FL framework for parameter aggregation.
    
    Args:
        model (nn.Module): PyTorch model to extract parameters from
        
    Returns:
        List[np.ndarray]: List of parameter arrays
    """
    return [param.detach().cpu().numpy() for param in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Set model parameters from list of numpy arrays from Flower.
    
    This function takes parameters received from the federated learning server
    (in numpy array format) and loads them into the PyTorch model.
    
    Args:
        model (nn.Module): PyTorch model to set parameters for
        parameters (List[np.ndarray]): List of parameter arrays from server
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def create_model(device: Optional[torch.device] = None) -> BreastCancerMLP:
    """
    Create and initialize a BreastCancerMLP model.
    
    Args:
        device (torch.device, optional): Device to move model to. If None, will use CUDA if available, else CPU.
    
    Returns:
        BreastCancerMLP: Initialized model on the specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BreastCancerMLP()
    model = model.to(device)
    
    return model


def model_to_flower_parameters(model: nn.Module) -> fl.common.Parameters:
    """
    Convert PyTorch model parameters to Flower Parameters format.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        fl.common.Parameters: Model parameters in Flower format
    """
    params_numpy = get_model_parameters(model)
    return fl.common.ndarrays_to_parameters(params_numpy)


def flower_parameters_to_model(model: nn.Module, parameters: fl.common.Parameters) -> None:
    """
    Load Flower Parameters into PyTorch model.
    
    Args:
        model (nn.Module): PyTorch model to load parameters into
        parameters (fl.common.Parameters): Parameters from Flower server
    """
    params_numpy = fl.common.parameters_to_ndarrays(parameters)
    set_model_parameters(model, params_numpy)

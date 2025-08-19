"""lunartech: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Normalize, ToTensor
import pandas as pd
import os, yaml, random
import numpy as np
from typing import Optional
from sklearn.model_selection import train_test_split

def set_seed_from_config(config_path):
    """Set global random seeds from a config.yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    seed = config['values']['RANDOM_SEED']
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed

class Net(nn.Module):
    def __init__(self, input_dim=30, dropout=0.2):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output layer
            nn.Linear(32, 1),
        )

        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml"))
        seed = set_seed_from_config(config_path)
        print(f"Random seed set to {seed}")


    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            self.X = torch.tensor(X.to_numpy(dtype=np.float32))
        else:
            self.X = torch.tensor(np.array(X, dtype=np.float32))
        if isinstance(y, (pd.DataFrame, pd.Series)):
            self.y = torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1, 1)
        else:
            self.y = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"feature": self.X[idx], "label": self.y[idx]}

def load_data(partition_id: int, num_partitions: int, batch_size: int = 32, data_dir: Optional[str] = None):
    """
    Load a data shard for a given partition_id and return (trainloader, valloader).
    Expects CSV files `hospital_a.csv` and `hospital_b.csv` with a 'target' column in repository data/ .
    """
    if data_dir is None:
        # Resolve repository data directory relative to this module
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

    shard_files = [
        os.path.join(data_dir, "hospital_a.csv"),
        os.path.join(data_dir, "hospital_b.csv"),
    ]

    # Map partition_id to a shard (supports num_partitions >= len(shard_files) by cycling)
    shard_idx = int(partition_id) % len(shard_files)
    df = pd.read_csv(shard_files[shard_idx])

    X = df.drop("target", axis=1)
    y = df["target"]

    # Local train/val split (85/15)
    stratify_arg = y if len(np.unique(y)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=1, shuffle=True, stratify=stratify_arg
    )

    train_ds = DictDataset(X_train, y_train)
    val_ds = DictDataset(X_val, y_val)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return trainloader, valloader

def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # Model to GPU
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            features = batch["feature"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / (len(trainloader) * max(1, epochs))
    return avg_trainloss

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            features = batch["feature"].to(device)
            labels = batch["label"].to(device)  # float (N,1)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            preds = (torch.sigmoid(outputs) >= 0.5).long().view(-1)  # (N,)
            true = labels.long().view(-1)                             # (N,)
            correct += (preds == true).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

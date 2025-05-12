
# models/chebnet_autoencoder.py

import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv

class ChebEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, K=2):
        super(ChebEncoder, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        self.conv2 = ChebConv(hidden_channels, in_channels, K=K)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class ChebAnomalyDetector(nn.Module):
    def __init__(self, encoder):
        super(ChebAnomalyDetector, self).__init__()
        self.encoder = encoder

    def forward(self, x, edge_index):
        x_hat = self.encoder(x, edge_index)
        loss = nn.functional.mse_loss(x_hat, x)
        return x_hat, loss

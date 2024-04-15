# Suppose 
import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn import TransformerConv, LSTMAggregation
from torch_geometric.utils import to_dense_batch, sort_edge_index

hidden_channels, num_classes = 0,3


class GAN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.6, edge_dim=None):
    
        self.conv1 = GATv2Conv(
            in_channels=in_channels, 
            out_channels=out_channels, 
            heads=heads, 
            concat=concat, 
            dropout=dropout,
            edge_dim=edge_dim
        )
        #self.lstm_agg = LSTMAggregation()
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        out = self.classifier(x)
        return out
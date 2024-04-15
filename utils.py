from torch_geometric.data import TemporalData, Dataset
import pandas as pd
import yfinance as yf
import torch

from data import *

# response = fetch(url)
# response. # get symbols
# stocks = ["SPY", "AMZN", "TSLA", "AAPL", "GOOGL", "META", "GM", "MS"]
# market = get_portfolio("SPY")


def train():
    # get model
    # get stock graph, sample graph
    # loop:
        # predict y, step, until len(x_pred)==0
        # backprop
    pass


def movement(x_pred, pct=0.03):
    high = max(x_pred[:,0])
    low = min(x_pred[:,1])

    if high >= 1 + pct:
        y = torch.tensor([1, 0, 0])  # above Pct return
    elif low <= 1 - pct:
        y = torch.tensor([0, 0, 1])  # below Pct return
    else:
        y = torch.tensor([0, 1, 0])  # within Pct return

    return y


def step(x_samp, x_pred):
    # Adds
    x_samp = torch.cat([x_samp, x_pred[:1]])
    x_pred = x_pred[1:]
    return x_samp, x_pred


def get_symbols(symbols):
    # symbol
    # stock graph
    # sample batches
    pass    


def download():
    # Downloads symbol data
    pass

def process():
    # Reads downloaded data
    pass
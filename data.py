from torch_geometric.data import TemporalData, Dataset, HeteroData
import yfinance as yf
import pandas as pd
import datetime as dt
import torch

from utils import *

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=730)
interval = '1h'


class Dataset(HeteroData):
    def __init__(self):
        self.symbols = None
        self.symbol_graphs = []

    def __getitem__(self):
        return 



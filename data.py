from torch_geometric.data import TemporalData, Dataset
import yfinance as yf
import pandas as pd
import datetime as dt
import torch

from utils import *

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=730)
interval = '30m'


class Dataset(HeteroData):
    def __init__(self):
        self.symbols = None
        self.symbol_graphs = []

    def __getitem__(self):
        return 


def make_graph(symbol):
    df = yf.download(symbol, start=startDate, end=endDate, interval=interval) # Portfolio
    df['NodeID'] = range(len(df))

    # Edges
    src = torch.tensor(df['NodeID'][:-1].values, dtype=torch.long) # [0,1,...,n-1]
    dst = torch.tensor(df['NodeID'][1:].values, dtype=torch.long) # [1,2,...,n]

    # Converts Datetime column to integer list
    datetime_str_list = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in df.index.tolist()][:-1] # ["Y-m-d H:M:S", ...]
    t = pd.to_datetime(datetime_str_list).astype('datetime64[s]').astype(int) # [160000, 160001, ...]
    
    # Price info
    msg_features = ['High', 'Low', 'Close', 'Open']
    msg = torch.tensor(df[msg_features][:-1].values, dtype=torch.float) # [High price, Low price, Close price, Open price]

    # Create graph
    return TemporalData(src=src, dst=dst, t=t, msg=msg)


def sample_graph(stock_graph, sample_length=13*30, pred_length=13*5): # 13 30m intervals in a trading day
    x = stock_graph.msg
    edge_index = torch.stack([stock_graph.src, stock_graph.dst], dim=0).reshape(-1,2)

    # Sample index
    ix = torch.randint(0, len(x) - sample_length - pred_length, (1,)) 
    curr_price_ix = ix + sample_length

    # Sample edges
    edge_samp = edge_index[ix:curr_price_ix+pred_length+1]

    # Sample time stamps
    t_samp = stock_graph.t[ix:curr_price_ix+pred_length+1]

    # Sample nodes
    x_samp = x[ix:curr_price_ix]
    last_close = float(x_samp[-1, 2]) 
    x_samp /= last_close # normalize by the LAST Close price

    # Prediction
    x_pred = x[curr_price_ix+1:curr_price_ix+pred_length+1] 
    x_pred /= last_close

    y = movement(x_pred) # [above Pct return, within Pct return, below Pct return] one hot

    return x_samp, x_pred, edge_samp, y
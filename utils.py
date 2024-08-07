from torch_geometric.data import HeteroData
import yfinance as yf
import torch

import numpy as np
import datetime as dt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.data import TemporalData, HeteroData



def sym_graph(symbol, interval='1h'):
  graph = HeteroData()
  df = yf.download(symbol, start=startDate, end=endDate, interval=interval)

  # Data
  msg_features = ['High', 'Low', 'Close', 'Open']
  msg = torch.tensor(df[msg_features][:-1].values, dtype=torch.float)
  t = df.index.strftime('%H%M%S').astype('int64')
  node_ids = torch.tensor(df.index.strftime('%Y%m%d%H%M%S').astype('int64'), dtype=torch.int64)
  #print(node_ids, msg, t)

  # Add data to graph
  graph.sym = symbol
  graph[symbol].x = msg
  graph[symbol].t = t
  graph[symbol].node_ids = node_ids

  # Create edges based on time sequence
  num_nodes = msg.size(0)
  src = torch.arange(0, num_nodes - 1, dtype=torch.long)  # Indices from 0 to N-2
  dst = torch.arange(1, num_nodes, dtype=torch.long)
  graph[symbol, 'next_in_sequence', symbol].edge_index = torch.stack([src, dst], dim=0)

  return graph


def build_graph(stock_symbols):
  graph = HeteroData()
  sp = sym_graph('SPY')
  stock_graphs = [sym_graph(sym) for sym in stock_symbols]

  # Add stock graphs to graph
  for s_graph in stock_graphs:
    graph[s_graph.sym].x = s_graph[s_graph.sym].x
    graph[s_graph.sym].t = s_graph[s_graph.sym].t
    graph[s_graph.sym].node_ids = s_graph[s_graph.sym].node_ids
    graph[s_graph.sym, 'next_in_sequence', s_graph.sym].edge_index = s_graph[s_graph.sym, 'next_in_sequence', s_graph.sym].edge_index

  # Add SPY stock graph
  graph[sp.sym].x = sp[sp.sym].x
  graph[sp.sym].t = sp[sp.sym].t
  graph[sp.sym].node_ids = sp[sp.sym].node_ids
  graph[sp.sym, 'next_in_sequence', sp.sym].edge_index = sp[sp.sym, 'next_in_sequence', sp.sym].edge_index

  # Link all symbol nodes to SPY nodes at the 'same_time' t
  sp_nodes = graph[sp.sym].node_ids
  for sym in stock_symbols:
    stock_nodes = graph[sym].node_ids
    graph[sp.sym, 'same_time', sym].edge_index = link_graphs(sp_nodes, stock_nodes)

  return graph


def link_graphs(sp_nodes, stock_nodes):
  # SPY/Stock look up node_id: idx
  stock_mapping = {node_id.item(): idx for idx, node_id in enumerate(stock_nodes)}
  sp_mapping = {node_id.item(): idx for idx, node_id in enumerate(sp_nodes)}

  common_node_ids = set(stock_mapping.keys()).intersection(sp_mapping.keys())

  # Makes list of common node id's idxs
  stock_indices = torch.tensor(sorted([stock_mapping[nid] for nid in common_node_ids]), dtype=torch.long) ###
  sp_indices = torch.tensor(sorted([sp_mapping[nid] for nid in common_node_ids]), dtype=torch.long) ###

  return torch.stack([sp_indices, stock_indices], dim=0)



def step(sample_graph, ix, sym):
  """
  Makes a time step through the sample_graph

  Updates:
    - x samp/pred
    - edge_index samp/pred
    - t samp/pred
    - node_ids samp/pred
  """
  # Update x_samp and x_pred
  sp_ix_raw, _ = sample_graph['SPY', 'same_time', sym].edge_index[:, ix]

  sample_graph[sym].x_samp = torch.cat([sample_graph[sym].x_samp, sample_graph[sym].x_pred[:1]])
  sample_graph[sym].x_pred = sample_graph[sym].x_pred[1:]

  sp_edge_indices = sample_graph['SPY', 'next_in_sequence', 'SPY'].edge_index[1]
  sp_ix = torch.where(sp_edge_indices == sp_ix_raw)[0].item()
  sp_len = sp_ix - sample_graph['SPY'].x_samp.shape[1]
  #print(len(sp_edge_indices), sp_ix_raw, sp_ix, sp_len)

  sample_graph['SPY'].x_samp = torch.cat([sample_graph['SPY'].x_samp, sample_graph['SPY'].x_pred[:sp_len]])
  sample_graph['SPY'].x_pred = sample_graph['SPY'].x_pred[sp_len:]

  # Update edge_index
  #

  print("Time step... ")
  return sample_graph, ix+1


def normalize(x, curr_ix, pred_ix):
  x_samp = x[:curr_ix]
  last_close = float(x_samp[-1, 2]) # Scale by last Close price, buy price

  x_samp /= last_close

  x_pred = x[curr_ix:pred_ix+1]
  x_pred /= last_close
  return x_samp, x_pred, last_close


def movement(x_pred, pct=0.03):
    high = max(x_pred[0])
    low = min(x_pred[1])

    if high >= 1 + pct:
        y = torch.tensor([1, 0, 0])  # above Pct return
    elif low <= 1 - pct:
        y = torch.tensor([0, 0, 1])  # below Pct return
    else:
        y = torch.tensor([0, 1, 0])  # within Pct return

    return y


#######################################################################
def sample(graph, sym, sample_len, pred_len, live):
  sample_graph = HeteroData()

  # random stock sample index
  s_ix = torch.randint(0, graph[sym].x.size(0) - sample_len - pred_len, (1,)) if not live else int(graph[sym].x.size(0) - sample_len - pred_len)
  curr_ix = s_ix+sample_len

  # stock graph
  sample_graph[sym].x = graph[sym].x[s_ix:curr_ix + pred_len]
  sample_graph[sym].t = graph[sym].t[s_ix:curr_ix + pred_len]
  sample_graph[sym].node_ids = graph[sym].node_ids[s_ix:curr_ix + pred_len]

  # stock nodesIDs
  first_nodeID = graph[sym].node_ids[s_ix]
  curr_nodeID = graph[sym].node_ids[curr_ix]
  last_nodeID = graph[sym].node_ids[curr_ix + pred_len]

  # spy nodes ixs
  f_sp_ix = torch.where(graph['SPY'].node_ids == first_nodeID.item())[0] # First
  c_sp_ix = torch.where(graph['SPY'].node_ids == curr_nodeID.item())[0] # Current
  l_sp_ix = torch.where(graph['SPY'].node_ids == last_nodeID.item())[0] # Last

  # spy graph
  sample_graph['SPY'].x = graph['SPY'].x[f_sp_ix:l_sp_ix]
  sample_graph['SPY'].t = graph['SPY'].t[f_sp_ix:l_sp_ix]
  sample_graph['SPY'].node_ids = graph['SPY'].node_ids[f_sp_ix:l_sp_ix]

  # sample edges
  sym_edges = graph[sym, 'next_in_sequence', sym].edge_index[:, s_ix:curr_ix + pred_len] ###
  spy_edges = graph['SPY', 'next_in_sequence', 'SPY'].edge_index[:, f_sp_ix:l_sp_ix] ###
  same_time_edges =  graph['SPY', 'same_time', sym].edge_index[:, s_ix:curr_ix + pred_len -1] ###

  dicts, edge_ixs = make_dicts([spy_edges, sym_edges]) # spy, sym
  same_time_edges = same_time_ix(same_time_edges, dicts) # convert same time ixs from graph ixs to sample graph ixs

  # set edges
  sample_graph['SPY', 'next_in_sequence', 'SPY'].edge_index = edge_ixs[0]
  sample_graph[sym, 'next_in_sequence', sym].edge_index = edge_ixs[1]
  sample_graph['SPY', 'same_time', sym].edge_index = same_time_edges

  # normalize stock data
  sample_graph[sym].x_samp, sample_graph[sym].x_pred, buy_price = normalize(sample_graph[sym].x, curr_ix=curr_ix - s_ix, pred_ix=curr_ix+pred_len - s_ix) # shifts ixs by first stock ix
  sample_graph['SPY'].x_samp, sample_graph['SPY'].x_pred, _ = normalize(sample_graph['SPY'].x, curr_ix=c_sp_ix - f_sp_ix, pred_ix=l_sp_ix - f_sp_ix) # shifts ixs by first spy stock ix

  sample_graph.buy_price = buy_price
  # EDGE_INDEX_SAMP, EDGE_INDEX_PRED
  #
  #

  # REMOVE REDUNDANT sample_graph.x #
  del sample_graph[sym].x
  del sample_graph['SPY'].x

  if live:
      return sample_graph, s_ix
  else:
      return sample_graph, len(sample_graph[sym].x_samp)
#######################################################################


def same_time_ix(same_time, dicts):
  edge_ix = torch.zeros_like(same_time) # same shape

  for i in range(same_time.shape[1]-1): ###
    edge_ix[0, i] = dicts[0][same_time[0, i].item()] # spy dict
    edge_ix[1, i] = dicts[1][same_time[1, i].item()] # sym dict

  #print(dicts, edge_ix)
  return edge_ix


def make_dicts(edge_sets):
  dicts = list()
  edge_ixs = list()

  for edge_samp in edge_sets:
    num_nodes = edge_samp.shape[1]

    edge_ix = torch.stack([torch.arange(0, num_nodes, dtype=torch.long),    # [[0, 1, 2, ...],
                          torch.arange(1, num_nodes+1, dtype=torch.long)], dim=0) #  [1, 2, 3, ...]]

    # dict of graph ixs to sample graph ixs
    gix_to_sgix = {edge_samp[j, i].item(): edge_ix[j, i].item()
                    for i in range(num_nodes) for j in range(2)} # [2, num_nodes]
    # gix_to_sgix[edge_samp[1, num_nodes-2].item()] = edge_ix[1, num_nodes-2].item()

    dicts.append(gix_to_sgix)
    edge_ixs.append(edge_ix)

  return dicts, edge_ixs # spy, sym



# acc = Account()
# acc.test()


# sym = 'TCRX'
# graph = build_graph(['AMZN', 'MSFT', 'TCRX'])
# sample_graph, ix = sample(graph, sym)
# print(graph.metadata())
# sample_graph

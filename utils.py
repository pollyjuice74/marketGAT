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


def build_graph(stock_symbols, startDate, endDate):
  graph = HeteroData()
  sp = sym_graph('SPY', startDate, endDate)
  stock_graphs = [sym_graph(sym, startDate, endDate) for sym in stock_symbols]

  # Add stock graphs to graph
  for s_graph in stock_graphs:
    graph[s_graph.sym].x = s_graph[s_graph.sym].x
    graph[s_graph.sym].t = torch.tensor(s_graph[s_graph.sym].t).unsqueeze(1)
    graph[s_graph.sym].node_ids = s_graph[s_graph.sym].node_ids
    graph[s_graph.sym, 'next_in_sequence', s_graph.sym].edge_index = s_graph[s_graph.sym, 'next_in_sequence', s_graph.sym].edge_index

  # Add SPY stock graph
  graph[sp.sym].x = sp[sp.sym].x
  graph[sp.sym].t = torch.tensor(sp[sp.sym].t).unsqueeze(1)
  graph[sp.sym].node_ids = sp[sp.sym].node_ids
  graph[sp.sym, 'next_in_sequence', sp.sym].edge_index = sp[sp.sym, 'next_in_sequence', sp.sym].edge_index

  # Link all symbol nodes to SPY nodes at the 'same_time' t
  sp_nodes = graph[sp.sym].node_ids
  for sym in stock_symbols:
    stock_nodes = graph[sym].node_ids
    graph[sp.sym, 'same_time', sym].edge_index = link_graphs(sp_nodes, stock_nodes)

  return graph
  

def sym_graph(symbol, startDate, endDate, interval='1h'):
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


def link_graphs(sp_nodes, stock_nodes):
  # SPY/Stock look up node_id: idx
  stock_mapping = {node_id.item(): idx for idx, node_id in enumerate(stock_nodes)}
  sp_mapping = {node_id.item(): idx for idx, node_id in enumerate(sp_nodes)}

  common_node_ids = set(stock_mapping.keys()).intersection(sp_mapping.keys())

  # Makes list of common node id's idxs
  stock_indices = torch.tensor(sorted([stock_mapping[nid] for nid in common_node_ids]), dtype=torch.long) ###
  sp_indices = torch.tensor(sorted([sp_mapping[nid] for nid in common_node_ids]), dtype=torch.long) ###

  return torch.stack([sp_indices, stock_indices], dim=0)



def step(sample_graph, curr_ixs):
  """
  Makes a time step through the sample_graph

  Updates:
    - x samp/pred
    - edge_index samp/pred
    - node_ids 
  """
  # update SPY data
  sample_graph['SPY'].x_samp = torch.cat([sample_graph['SPY'].x_samp, sample_graph['SPY'].x_pred[:1]])
  sample_graph['SPY'].x_pred = sample_graph['SPY'].x_pred[1:]

  # update SPY next_in_sequence edges
  sample_graph['SPY', 'next_in_sequence', 'SPY'].edge_index_samp = torch.cat([sample_graph['SPY', 'next_in_sequence', 'SPY'].edge_index_samp,
                                                                              sample_graph['SPY', 'next_in_sequence', 'SPY'].edge_index_pred[:, :1]], dim=1)
  sample_graph['SPY', 'next_in_sequence', 'SPY'].edge_index_pred = sample_graph['SPY', 'next_in_sequence', 'SPY'].edge_index_pred[:, 1:]

  # SPY current ix
  spy_ix = curr_ixs['SPY'] 

  for sym, _ in curr_ixs.items():
    if sym == 'SPY':
      continue

    # get corresponding SPY-stock same_time node ix if any
    spy_pred_edges = sample_graph['SPY', 'same_time', sym].edge_index_pred[0]
    sym_ix = torch.where(spy_pred_edges == spy_ix)[0].item() 

    # update SPY-stock same_time edges
    sample_graph['SPY', 'same_time', sym].edge_index_samp = torch.cat([sample_graph['SPY', 'same_time', sym].edge_index_samp, 
                                                                       sample_graph['SPY', 'same_time', sym].edge_index_pred[:, :sym_ix]], dim=1)
    sample_graph['SPY', 'same_time', sym].edge_index_pred = sample_graph['SPY', 'same_time', sym].edge_index_pred[:, sym_ix:]

    # update stock next_in_sequence edges 
    sample_graph[sym, 'next_in_sequence', sym].edge_index_samp = torch.cat([sample_graph[sym, 'next_in_sequence', sym].edge_index_samp,
                                                                              sample_graph[sym, 'next_in_sequence', sym].edge_index_pred[:, :sym_ix]], dim=1)
    sample_graph[sym, 'next_in_sequence', sym].edge_index_pred = sample_graph[sym, 'next_in_sequence', sym].edge_index_pred[:, sym_ix:]

    # update stock data
    sample_graph[sym].x_samp = torch.cat([sample_graph[sym].x_samp, sample_graph[sym].x_pred[:sym_ix]])
    sample_graph[sym].x_pred = sample_graph[sym].x_pred[sym_ix:]

  # update spy_ix
  curr_ixs['SPY'] += 1

  print("Time step... ")
  return sample_graph, curr_ixs


def normalize(x, curr_ix, pred_ix):
  x_samp = x[:curr_ix]
  last_close = float(x_samp[-1, 2]) # Scale by last Close price, buy price

  x_samp /= last_close

  x_pred = x[curr_ix:pred_ix+1]
  x_pred /= last_close
  return x_samp, x_pred, last_close


def make_time_dict(sample_graph, stock_symbols):
    time_dict = dict()

    # initialize zeros ix
    for sym in stock_symbols:
      print(sym)
      print(sample_graph[sym].x_pred[:, 0])
      time_dict[sym] = torch.zeros(sample_graph[sym].x_pred[:, 0].shape)

    return time_dict


def movement(sample_graph, max_threshold=0.30, interval=0.01):
    stock_symbols = [key for key in sample_graph.node_types if key.isupper()]
    pcts = [round(i * interval, 2) for i in range(1, int(max_threshold / interval) + 1)]
    thresholds = pcts + [-p for p in pcts]
    thresholds.sort(reverse=True) # desc order pct changes intervals

    # predictions
    y = dict() # sym -> one-hot pred
    time_dict = make_time_dict(sample_graph, stock_symbols) # initialized x's to all zero tensor

    for sym in stock_symbols:
      # print(sample_graph[sym].x_pred[:, 0])
      high, high_ix = torch.max(sample_graph[sym].x_pred[:, 0], dim=0) # high of the x_pred data
      # low, low_ix = torch.min(sample_graph[sym].x_pred[:, 1], dim=0) # high of the x_pred data

      time_dict[sym][high_ix] = 1 # fills one-hot high_ix of high pct chg

      y[sym] = torch.zeros( len(thresholds)+1 )

      # assigns one-hot high pos to y tensor
      for ix, pct in enumerate(thresholds):
        if high >= 1 + pct:
          y[sym][ix] = 1
          break

    return y, time_dict


#######################################################################
def sample(graph, symbols, sample_len, pred_len, live):
    sample_graph = HeteroData()

    while True:
        try:  
            # random SPY sample index
            spy_ix = torch.randint(0, graph['SPY'].x.size(0) - sample_len - pred_len, (1,)) if not live else int(graph['SPY'].x.size(0) - sample_len - pred_len)
            curr_ix = spy_ix + sample_len

            # print( graph['SPY'].t.shape)
            # SPY graph
            sample_graph['SPY'].x = torch.cat([graph['SPY'].x[spy_ix:curr_ix + pred_len],
                                              graph['SPY'].t[spy_ix:curr_ix + pred_len]], dim=1)
            sample_graph['SPY'].node_ids = graph['SPY'].node_ids[spy_ix:curr_ix + pred_len]

            # SPY nodesIDs
            first_nodeID = graph['SPY'].node_ids[spy_ix]
            curr_nodeID = graph['SPY'].node_ids[curr_ix]
            last_nodeID = graph['SPY'].node_ids[curr_ix + pred_len]

            # SPY edges
            spy_edges = graph['SPY', 'next_in_sequence', 'SPY'].edge_index[:, spy_ix:curr_ix + pred_len] ###

            for sym in symbols:
                if sym == 'SPY':
                  continue

                # stock nodes ixs
                f_sym_ix = torch.where(graph[sym].node_ids == first_nodeID.item())[0] # First
                c_sym_ix = torch.where(graph[sym].node_ids == curr_nodeID.item())[0] # Current
                l_sym_ix = torch.where(graph[sym].node_ids == last_nodeID.item())[0] # Last

                # stock graph
                sample_graph[sym].x = torch.cat([graph[sym].x[f_sym_ix:l_sym_ix],
                                                graph[sym].t[f_sym_ix:l_sym_ix]], dim=1)
                sample_graph[sym].node_ids = graph[sym].node_ids[f_sym_ix:l_sym_ix]

                # stock edges
                sym_edges = graph[sym, 'next_in_sequence', sym].edge_index[:, f_sym_ix:l_sym_ix] ###
                same_time_edges =  graph['SPY', 'same_time', sym].edge_index[:, f_sym_ix:l_sym_ix -1] ###

                dicts, edge_ixs = make_dicts([spy_edges, sym_edges]) # spy, sym
                same_time_edges = same_time_ix(same_time_edges, dicts) # convert same time ixs from graph ixs to sample graph ixs
                # print(same_time_edges)

                # set edges 
                sample_graph[sym, 'next_in_sequence', sym].edge_index = edge_ixs[1]
                sample_graph['SPY', 'same_time', sym].edge_index = same_time_edges

                # normalize stock data and split into sample and pred
                sample_graph[sym].x_samp, sample_graph[sym].x_pred, buy_price = normalize(sample_graph[sym].x, curr_ix=c_sym_ix - f_sym_ix, pred_ix=l_sym_ix - f_sym_ix) # shifts ixs by first stock ix
              
                sample_graph.buy_price = buy_price

                # split edges into sample and pred
                sample_graph[sym, 'next_in_sequence', sym].edge_index_samp = sample_graph[sym, 'next_in_sequence', sym].edge_index[:, :c_sym_ix - f_sym_ix]
                sample_graph[sym, 'next_in_sequence', sym].edge_index_pred = sample_graph[sym, 'next_in_sequence', sym].edge_index[:, c_sym_ix - f_sym_ix:l_sym_ix - f_sym_ix+1]
                
                sample_graph['SPY', 'same_time', sym].edge_index_samp = sample_graph['SPY', 'same_time', sym].edge_index[:, :c_sym_ix - f_sym_ix]
                sample_graph['SPY', 'same_time', sym].edge_index_pred = sample_graph['SPY', 'same_time', sym].edge_index[:, c_sym_ix - f_sym_ix:l_sym_ix - f_sym_ix+1]
                # print(sample_graph[sym, 'next_in_sequence', sym].edge_index_samp)

            # set SPY edges, normalize SPY data and split into sample and pred
            sample_graph['SPY', 'next_in_sequence', 'SPY'].edge_index = edge_ixs[0]
            sample_graph['SPY'].x_samp, sample_graph['SPY'].x_pred, _ = normalize(sample_graph['SPY'].x, curr_ix=curr_ix - spy_ix, pred_ix=curr_ix+pred_len - spy_ix) # shifts ixs by first spy stock ix

            # split 'next_in_sequence' edges into sample and pred
            sample_graph['SPY', 'next_in_sequence', 'SPY'].edge_index_samp = sample_graph[sym, 'next_in_sequence', sym].edge_index[:, :curr_ix - spy_ix]
            sample_graph['SPY', 'next_in_sequence', 'SPY'].edge_index_pred = sample_graph[sym, 'next_in_sequence', sym].edge_index[:, curr_ix - spy_ix:curr_ix+pred_len - spy_ix + 1]
            # print(sample_graph['SPY'].x_pred)

            # REMOVE REDUNDANT sample_graph.x #
            del sample_graph['SPY'].x
            del sample_graph['SPY', 'next_in_sequence', 'SPY'].edge_index

            for sym in symbols:
              if sym == 'SPY':
                continue

              del sample_graph[sym].x
              del sample_graph[sym, 'next_in_sequence', sym].edge_index
              del sample_graph['SPY', 'same_time', sym].edge_index

            if live:
                return sample_graph, spy_ix
            else:
                curr_ixs = {sym: len(sample_graph[sym].x_samp) for sym in symbols}
                return sample_graph, curr_ixs

        except Exception as e:
            print(f"Retrying due to error: {e}")
            continue
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

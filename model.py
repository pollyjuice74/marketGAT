# Suppose 
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

import torch_geometric
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn import TransformerConv, LSTMAggregation
from torch_geometric.utils import to_dense_batch, sort_edge_index

from utils import *

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.utils.hetero import construct_bipartite_edge_index


hidden_channels, num_classes = 0,3
in_channels, out_channels = 0, 0



class Stock:
    def __init__(self, sym, ammount, train=True):
        self.price = self.get_price(sym) # setted with live_price
        self.total = ammount * self.live_price(sym) if train else ammount * self.price(sym)

        self.graph = build(sym)


    def live_price(self, sym):
        pass


    def price(self, sym):
        pass


    def build(self, sym):
        pass



class HGTConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(metadata[1])
        }

        self.dst_node_types = set([key[-1] for key in self.edge_types])

        self.kqv_lin = HeteroDictLinear(self.in_channels,
                                        self.out_channels * 3)

        self.out_lin = HeteroDictLinear(self.out_channels, self.out_channels,
                                        types=self.node_types)

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)

        self.k_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)

        self.skip = ParameterDict({
            node_type: Parameter(torch.empty(1))
            for node_type in self.node_types
        })

        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_type = '__'.join(edge_type)
            self.p_rel[edge_type] = Parameter(torch.empty(1, heads))

        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        ones(self.skip)
        ones(self.p_rel)


    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """Concatenates a dictionary of features."""
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset


    def _construct_src_node_feat(
        self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs the source node representations."""
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads

        # Flatten into a single tensor with shape [num_edge_types * heads, D]:
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType] = {}
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            # construct type_vec for curr edge_type with shape [H, D]
            edge_type_offset = self.edge_types_map[edge_type]
            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(
                1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        type_vec = torch.cat(type_list, dim=1).flatten()

        k = self.k_rel(ks, type_vec).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec).view(H, -1, D).transpose(0, 1)

        return k, v, offset


    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]  # Support both.
    ) -> Dict[NodeType, Optional[Tensor]]:
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Compute K, Q, V over node types:
        kqv_dict = self.kqv_lin(x_dict)
        print(kqv_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict)

        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel,
            num_nodes=k.size(0))

        print(edge_index.shape, k.permute(2, 1, 0).shape, q.shape, v.permute(2, 1, 0).shape, edge_attr.shape)
        print(edge_index)
        out = self.propagate(edge_index, k=k.permute(2, 1, 0), q=q, v=v.permute(2, 1, 0), edge_attr=edge_attr)

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings:
        a_dict = self.out_lin({
            k:
            torch.nn.functional.gelu(v) if v is not None else v
            for k, v in out_dict.items()
        })

        # Iterate over node types:
        for node_type, out in out_dict.items():
            out = a_dict[node_type]

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict


    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')



class HGT(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, num_classes=3, heads=1, num_layers=1):
        super().__init__()

        # node_type: linear layer
        self.lin_dict = torch.nn.ModuleDict()

        # List of HGTConv layers
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
          conv = HGTConv(
              in_channels=hidden_channels,
              out_channels=hidden_channels,
              metadata=metadata,
              heads=heads,
          )
          self.convs.append(conv)

        self.classifier = nn.Linear(hidden_channels, num_classes)


    def forward(self, x_dict, edge_index_dict, edge_attr=None):
        # print([v.shape for k, v in x_dict.items()])
        # print(edge_index_dict)

        for node_type, x in x_dict.items():
            out_channels = x.size(0)
            layer = Linear(out_channels, 245)
            x_dict[node_type] = layer(x.T).relu()

        print([v.shape for k, v in x_dict.items()])
        #print(x_dict)

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.classifier(x_dict)



class Account:
    def __init__(self, hidden_channels=7*35, epochs=2000, learning_rate=1e-05):
        # Account info
        self.net_value = 416.29
        self.symbols = ['AMZN',] #"TSLA", "AAPL", "GOOGL", "META", "GM", "MS"]
        self.bets = 100 # Limited ammount of bets per day
        # Bets
        self.current_bets = {sym: 0 for sym in self.symbols}
        self.stocks = [] # Graphs of stocks
        # Data
        self.graph = build_graph(self.symbols)
        # Model
        self.model = HGT(metadata=self.graph.metadata(), hidden_channels=hidden_channels)
        # Training
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        ##################################################


    def train(self):
        for i in range(self.epochs):
            epoch_loss = 0
            for sym in self.symbols:
                print("\nTraining on: ", sym)

                for _ in range(1000): # sample 1000 times
                    sample_graph, ix = sample(self.graph, sym) # Samples Training graph nodes
                    pct = torch.std(sample_graph[sym].x_samp) * 0.5 # try to predict a half std movement
                    print("Percent pred: ", pct.item())

                    while sample_graph[sym].x_pred.numel() > 0: # while there is more data to predict

                        y_hat = self.model(sample_graph.x_samp_dict, sample_graph.edge_index_dict)
                        y = movement(sample_graph[sym].x_pred[0], pct=pct.item())

                        loss = self.criterion(y_hat, y)
                        step(sample_graph, ix, sym)

                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        self.optimizer.step()

                epoch_loss += loss.item()

                self.print(i, epoch_loss, freq=25)




    def test(self, graph, live=False): # Live trading simulation
        for sym in self.symbols:
            x, edge_index = sample_graph_live(graph, sym) if live else sample_graph(graph, sym)# Samples LIVE graph nodes
            pred = self.model(x, edge_index, pct=0.03)

            if self.buy_conditions(pred):
                ammount = self.get_portion()
                self.buy(sym, ammount)
        
        self.wait()
        self.sell(sym, ammount)


    def get_portion(self):
        pass


    def buy_conditions(self):
        pass


    def print(self, i, epoch_loss, freq=25):
        if i % freq == 0:
            print(f"Epoch: {i+1}/{self.epochs}, Loss: {epoch_loss}")


    def buy(self, sym, ammount):
        self.current_bets[sym] += ammount # sym: ammount of stocks owned

        stock = Stock(sym, ammount)
        self.net_value -= ammount * stock.price
        
        self.stocks.append(stock)


    def wait(self):
        time.sleep(10)


    def sell(self, stock, sym, ammount):
        self.current_bets[sym] -= ammount # sym: ammount of stocks owned
        self.net_value += ammount * stock.price
        
        self.stocks.remove(stock)

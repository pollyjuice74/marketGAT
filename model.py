import torch
from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer, ModuleList, MultiheadAttention, Linear, Dropout
from torch_geometric.nn import GATv2Conv, to_hetero


class MarketTransformer(Module):
    def __init__(self, metadata, hidden_dims=5*16, num_classes=3, num_heads=8, num_layers=2):
        super().__init__()
        assert  hidden_dims % num_heads == 0, "Hidden dims and num_heads don't match"
        self.embedding = torch.nn.Linear(5, hidden_dims) # learned h,l,o,c,time (hh:mm:ss) representations
        self.pos_encoder = PositionalEncoder(hidden_dims) # retain temporal order info
        self.self_attn = MultiheadAttention(hidden_dims, num_heads) # each stock will attend to itself

        self.gat_convs = GATModule(hidden_dims, num_heads, num_layers, metadata[1]) # capturing graph structure properties
        self.gat_convs = to_hetero(self.gat_convs, metadata, aggr="sum") # turn homogeneous gnn into hetero gnn

        # for capturing relations between different stocks over time
        encoder_layers = TransformerEncoderLayer(d_model=hidden_dims, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

        # one-hot prediciton of stock's movement
        self.classifier = Linear(hidden_dims, num_classes)
        # time ix prediction of when the stock will reach pos. returns
        self.time_prediction = Linear(hidden_dims, 1)


    def forward(self, x_dict, edge_index_dict):
        for key, x in x_dict.items():
            x = self.embedding(x)
            x = self.pos_encoder(x)
            x_dict[key], _ = self.self_attn(x,x,x)

        x_dict = self.gat_convs(x_dict, edge_index_dict)

        x, mask = self._to_transformer_input(x_dict)
        x = self.transformer_encoder(x, mask)

        class_out_dict, time_out_dict = dict(), dict()
        for key in x_dict.keys():
            class_out_dict[key] = self.classifier(x)
            time_out_dict[key] = self.time_prediction(x)

        return out_dict

    def _to_transformer_input(self, x_dict):
        ### TODO
        all_x, mask = [], []
        return x, mask

    def compute_utility(class_out_dict, time_out_dict):
        utility_dict = dict()
        for key in class_out_dict.keys():
            P_return = torch.softmax(class_out_dict[key])
            utility_dict[key] = P_return / time_out_dict[key]
        return utility_dict

    def select_top_stocks():
        pass


class GATModule(Module):
    def __init__(self, hidden_dims, num_heads, num_layers, known_edge_types):
        super().__init__()
        self.gat_convs = ModuleList() # utilize graph structure of the data
        self.known_edge_types = known_edge_types
        for _ in range(num_layers):
            self.gat_convs.append( GATv2Conv(hidden_dims,
                                             hidden_dims//num_heads,
                                             heads=num_heads, add_self_loops=False) )

    def forward(self, x_dict, edge_index_dict):
        for conv in self.gat_convs:
          x_updated_dict = dict() # store dst node updates of x
          
          print(self.known_edge_types)
          for edge_type in self.known_edge_types:
            edge_index = edge_index_dict[edge_type]
            print(edge_type)
            src_node_type, _, dst_node_type = edge_type # tuple of (src_node_type, edge_type, dst_node_type)

            # passes src node data to GAT layer
            if src_node_type == dst_node_type:
              # next_in_sequence edges
              x_updated = conv(x_dict[src_node_type], edge_index)
            else:
              # same_time edges
              x_updated = conv((x_dict[src_node_type], x_dict[dst_node_type]), edge_index)

            # initialize/add x updates to dst node
            if dst_node_type in x_updated_dict:
              x_updated_dict[dst_node_type] += x_updated
            else:
              x_updated_dict[dst_node_type] = x_updated

        return x_updated_dict


class PositionalEncoder(Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(1e5)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term) # Apply sin to even indices in the array (2i)
        pe[:, 1::2] = torch.cos(position * div_term) # Apply cos to odd indices in the array (2i+1)

        self.register_buffer('pe', pe)
        self.dropout = Dropout(0.1)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# (s, p)->(s, p/m + f/v, k+u, d)->(s, k+u)->(s, u)->(s,1)->(1,)

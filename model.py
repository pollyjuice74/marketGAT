import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer, ModuleList, MultiheadAttention, Linear, Dropout
from torch_geometric.nn import GATv2Conv


class MarketTranformer(nn.Module):
        def __init__(self, metadata, hidden_dims=4*16, num_classes=3, num_heads=8, num_layers=2):
                super().__init__()
                self.embedding = torch.nn.Linear(4, hidden_dims) # learned h,l,o,c representations
                self.pos_encoder = PositionalEncoding(hidden_dim) # retain temporal order info
                self.self_attn = MultiheadAttention(hidden_dims, num_heads)        

                self.gat_conv = ModuleList() # utilize graph structure of the data
                for _ in range(num_layers):
                        self.gat_layers.append( GATv2Conv(hidden_dims, hidden_dims//num_heads, heads=num_heads) )

                # for capturing relations between different stocks over time
                encoder_layers = TransformerEncoderLayer(d_model=hidden_dims, nhead=num_heads)
                self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
                # one-hot prediciton of stock's movement
                self.classifier = Linear(hidden_dims, num_classes) 

                # turn homogeneous gnn into hetero gnn
                self.to_hetero(self.gat_layers, metadata, aggr="sum")
                                                 
        def forward(self, x_dict, edge_index_dict):
                for key, x in x_dict.items():
                        x = self.embedding(x)
                        x = self.pos_encoder(x)
                        x_dict[key], _ = self.self_attn(x,x,x)

                for conv in self.gat_conv:
                        x_dict = conv(x_dict, edge_index_dict)

                x, mask = self._to_transformer_input(x_dict)
                x = self.transformer_encoder(x, mask)

                out_dict = dict()
                for key in x_dict.keys():
                        out_dict[key] = self.classifier(x)

                return out_dict

        def _to_transformer_input(self, x_dict):
                all_x, mask = [], []
                return x, mask
                

class PositionalEncoder(nn.Module):
        def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp( torch.arange(0, d_model, 2).float() * (-torch.log( torch.tensor(1e5) ) / d_model) )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer( 'pe', pe.unsqueeze(0).transpose(0,1) )  
                
                self.dropout = Dropout(0.1)

        def forward(self, x):
                x = x + pe[:x.size(0), :]
                return self.dropout(x)
        

# (s, p)->(s, p/m + f/v, k+u, d)->(s, k+u)->(s, u)->(s,1)->(1,)  

class MarketNN(nn.Module):
        def __init__(self):
                self.mask = mask
                
        def forward(self, graph):
            # (s, p)->(s, p/m + f/v, k+u, d)->(s, k+u)->(s, u)->(s,1)->(1,)  
            emb_symbols = [ emb_sym(sym_graph.x) for sym_graph in graph ] 


class PositionalEncoder(nn.Module):
        

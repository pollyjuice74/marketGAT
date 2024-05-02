# marketNN
This repo is implementing a Graph Neural Network using PyTorch Geometric to predict market activity in a short period of time. 

## Contents:
`model.py`: Contains Account object for simulating trading, HGT model, HGTConv layer, Stock object to simulate stock possesion.
`utils.py`: Contains functions to create the data, main functs include `build_graph()` which builds the graph of stocks, `sample()` samples stock to SPY market index, `step()` steps to simulate trading by adding new time nodes. 
`data.py`: Other


### Stocks
Essentially each **stock** is visualized as a **linked graph of nodes** connected through time each node contains the **High, Low, Close, Open prices** for a given time interval (30m, 1h, etc.)

### Graph

SP500 index is sampled 

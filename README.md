# marketNN
This repo is implementing a Graph Neural Network using PyTorch Geometric to predict market activity in a short period of time. 

## Contents:
`model.py`: Contains Account object for simulating trading, HGT model, HGTConv layer, Stock object to simulate stock possesion.

`utils.py`: Contains functions to create the data, main functs include `build_graph()` which builds the graph of stocks, `sample()` samples stock to SPY market index, `step()` steps to simulate trading by adding new time nodes. 

`data.py`: Other

The thought process:
---

Essentially each **stock** is visualized as a **linked graph of nodes** connected through time each node contains the **High, Low, Close, Open prices** for a given time interval (30m, 1h, etc.). 

The graph can be visualized as many of these stock graphs or strings of a stock's price tape history as a time dimention. To this we add a second stock parallel to it with it's own price tape history, if we continue this process we could imagine a two dimentional plane where the x dimention is time, all stocks having the same time dimention, the y dimention is the individual stock's price history and the z dimention as the highs and lows of the y dimention's price history. Thus a (x, y, z) dimentional object.

In this specific implementation we imagine two strings, one a **stock** to sample and train/predict on, and parallel the **SP500 index**. The idea is to link these two stocks by time, so that each node in the stock's price history has a corresponding SP500 node that is the price of the index at the exact same time. 



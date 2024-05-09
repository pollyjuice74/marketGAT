# marketNN
This repo is implementing a Graph Neural Network using PyTorch Geometric to predict market activity in a short period of time. 

## Contents:
`marketGPT.ipynb`: Sandbox for tinkering and designing with the data and models.

`model.py`: Contains HGT model, HGTConv layer.

`utils.py`: Contains functions to create the data, main functs include `build_graph()` which builds the graph of stocks, `sample()` samples stock to SPY market index, `step()` steps to simulate trading by adding new time nodes. 

`acc.py`: Contains Account object for simulating trading and Stock object to simulate stock possesion.

The thought process:
---

Essentially each **stock** is visualized as a **linked graph of nodes** connected through time, each node contains the **High, Low, Close, Open prices** for a given time interval (30m, 1h, etc.). 

The graph can be visualized as many of these stock graphs or strings of a stock's price tape history as a time dimention. To this we add a second stock parallel to it with it's own price tape history, if we continue this process we could imagine a two dimentional plane where the x dimention is time, all stocks having the same time dimention, the y dimention is the individual stock's price history. Adding a third z dimention as the highs and lows of the y dimention's price history. Thus a (x, y, z) dimentional object.

In this specific implementation we imagine two strings, one a **stock** to sample and train/predict on, and parallel the **SP500 index**. The idea is to link these two stocks by time, so that each node in the stock's price history has a corresponding SP500 node that is the price of the index at the exact same time. 

Supposing that the graph of a stock and the SP500 are 760 days long at 1hr intervals. Sampling 35 days of data splitting it to 30 days of 'history' and 5 days of 'prediction'  and then predicting a x sized percent change, say a 1.03 change in the stock price from the 'starting' simulated buy price at day 30. We normalize the data by this 'buy price'. 

The `Account` is trying to predict weather given the 30 day history data how likely is the stock to make a x percent sized move and if so sell it and reap the profit. But if it doesn't, sell it at the end of the 5 days at whatever price. Giving the Account an incentive to be profitable, also the Account will have a portfolio and a specified portion that it can invest per trade. Thus spreading out gains and losses equitably and creating a sort of Portfolio. 

Later on I would like to create a Portfolio model that could optimize the portions and risks to be better able to identify a good stock and if so invest in it accordingly. 



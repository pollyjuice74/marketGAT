# marketNN
This repo is implementing a Graph Neural Network using PyTorch Geometric to predict market activity in a short period of time. 

For a mathematically rigorous explanation, see this paper by clicking [here](Probabilistic_Formalization_of_a_High_Volatility_Stock_Trading_Strategy_with_Automation_Potential.pdf) or the `Probabilistic_Formalization_of_a_High_Volatility_Stock_Trading_Strategy_with_Automation_Potential.pdf` file above.

**This project is still a working progress**

## Results

### Transformer NN Arquitecture and Data Processing
The model uses attention in three steps. The first uses Positional Encoding and Multiheaded Attention so that each stock can attend to it's own past data. The second uses a Graph 
Attention Transformer module that captures dependencies in the cylinder-like structure of the custom graph data where the anchor or middle stock's time sequence is the SPY index serving as a market average and/or baseline for comparing the movement between all other stocks. The third is a Transformer Encoder layer that captures inter stock dependencies over time. 

The predictions are taken from a split of say 35 days of data, 30 are history and 5 are prediciton time days. They are splitted two one-hot vector predictions, one predicting at what time if a stock hit a desired percent change above the bought price and another for when the high of the predicted sample occurred. The model updates it's predictions based on incoming data simulating the passage of time until there are no more prediction days available and it would have to sell at whatever price is at the market. This is similar to how options expire, giving the model more pressure to be profitable in a short amount of time.

The model is still subject to change based on my recent development on formalizing my stock trading strategy in the paper above based on observing local maximas and minimas in the stock's price history.

---

### Graph Structure
Here we can see how the cylinder-like structured data is represented as a graph, this is a sample of 245 time intervals from a much larger sample of the market taken at random so that there is no memorization of the market and the model learns to predict short movements based on limited historical data.

We can see 3 stocks that are anchored to the `SPY` index and the data is split between x_samp and x_pred. A stock's historical data is linked with `next_in_sequence` edges and all stocks are linked at their corresponding times to the `SPY` linked graph with the `same_time` edges. 

This can be found in the `marketGPT.ipynb` file. 

```
HeteroData
  SPY={
    node_ids=[245],
    x_samp=[210, 5],
    x_pred=[35, 5],
  },
  TBLA={
    node_ids=[245],
    x_samp=[210, 5],
    x_pred=[35, 5],
  },
  MS={
    node_ids=[245],
    x_samp=[210, 5],
    x_pred=[35, 5],
  },
  PBM={
    node_ids=[239],
    x_samp=[204, 5],
    x_pred=[35, 5],
  },


  (TBLA, next_in_sequence, TBLA)={
    edge_index_samp=[2, 210],
    edge_index_pred=[2, 35],
  },
  (SPY, same_time, TBLA)={
    edge_index_samp=[2, 210],
    edge_index_pred=[2, 34],
  },
  (MS, next_in_sequence, MS)={
    edge_index_samp=[2, 210],
    edge_index_pred=[2, 35],
  },
  (SPY, same_time, MS)={
    edge_index_samp=[2, 210],
    edge_index_pred=[2, 34],
  },
  (PBM, next_in_sequence, PBM)={
    edge_index_samp=[2, 204],
    edge_index_pred=[2, 35],
  },
  (SPY, same_time, PBM)={
    edge_index_samp=[2, 204],
    edge_index_pred=[2, 34],
  },
 (SPY, next_in_sequence, SPY)={
    edge_index_samp=[2, 210],
    edge_index_pred=[2, 35],
  }
)
```

## Contents:
`marketGPT.ipynb`: Sandbox for tinkering and designing with the data and models.

`model.py`: Contains the implementation of the above mentioned neural network arquitecture.

`acc.py`: Contains Account object for simulating trading and Stock object to simulate stock possesion.

`utils.py`: Contains functions to create the data, main functs include `build_graph()` which builds the graph of stocks, `sample()` samples stock to SPY market index, `step()` steps to simulate trading by adding new time nodes and `movement()` that creates the model's predictions. 


## TODO:

- Incorporate a probabilistic approach to the NN model based on the paper above.

- Connect it to a live trading account (TradingView perhaps)

---



## Resources

- Four Models of Competiiton

https://journals.sagepub.com/doi/pdf/10.1177/0256090919940101 


-  Market-Guided Stock Transformer for Stock Price Forecasting

https://ar5iv.labs.arxiv.org/html/2312.15235

- Transformer-based deep learning model for stock return forecasting

https://www.utupub.fi/bitstream/handle/10024/154307/Paivarinta_Markus_ProGradu.pdf?sequence=1#:~:text=URL%3A%20https%3A%2F%2Fwww.utupub.fi%2Fbitstream%2Fhandle%2F10024%2F154307%2FPaivarinta_Markus_ProGradu.pdf%3Fsequence%3D1%0AVisible%3A%200%25%20

- Modeling Relational Data with Graph Convolutional Networks

https://arxiv.org/pdf/1703.06103

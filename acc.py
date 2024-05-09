# Suppose 
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

from model import *
from utils import *



class Stock:
    def __init__(self, sym, train=True):
        # Metadata
        self.sym = sym
        self.pct = None
        self.price = self.get_price() # setted with live_price
        # Graph
        self.sample_graph = None # HeteroData graph object
        ########################


    def update(self):
        """ Updates """
        self.price = self.get_price()


    def get_price(self, live=True):
        """ 
        Gets either the live price or the price at the current step
        of the sample_graph data object.
        """
        return self.live_price()[0][2] if live else self.price() 


    def live_price(self, interval='1m'):
        """ Gets info and sets live price of stock """
        s = yf.Ticker(self.sym)
        df = s.history(period='1d', interval=interval).iloc[-1]

        self.price = float(df['Close'].item()) # CLOSE price set as LIVE PRICE

        x =  torch.tensor(df[['High', 'Low', 'Close', 'Open']])
        t = df.index.strftime('%H%M%S').astype('int64')
        node_ids = torch.tensor(df.index.strftime('%Y%m%d%H%M%S').astype('int64'), dtype=torch.int64)

        return (x, t, node_ids)  # x,t,node_ids,edge_index
    

    def price(self):
        """ Get price from data """
        return self.sample_graph[self.sym].x_samp[2].item()




class Account:
    def __init__(self, hidden_channels=7*30, epochs=2000, learning_rate=1e-05):
        # Account info
        self.history = list() # Tape of transaction history
        self.net_value = 416.29
        self.portions = None # [1, 8]
        self.used = None # [1, 8]
        self.bets = 2 # Limited ammount of bets per day
        # Bets
        self.current_bets = {sym: [] for sym in self.symbols} # sym: [(amount in $, shares), ..., ]
        self.stocks = {sym: None for sym in self.symbols} # Graphs of stocks
        self.trailing_prices = None
        # Data
        self.symbols = ['AMZN',] #"TSLA", "AAPL", "GOOGL", "META", "GM", "MS"]
        self.graph = build_graph(self.symbols)
        self.sample_len = 7*30
        self.pred_len = 7*5
        # Model
        self.model = HGT(metadata=self.graph.metadata(), hidden_channels=hidden_channels)
        # Training
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        ##################################################


    def test(self, graph, live=False): # Live trading simulation
        """
        Simulates the model predicting movement of the stocks given a sample interval 
        and awaits for it to sell before beggining the process again.

        Goes in sequential order of a larger sample interval of a stock's history 
        and gives a profitablility calculation based on model predictions.
        """
        while True:

            # Checks for buying conditions in all symbols
            for sym in self.symbols:

                stock = self.stocks[sym]
                sample_graph, _ = self.sample_live(graph, sym) if live else sample(graph, sym, sample_len=self.sample_len, pred_len=self.pred_len)# Samples LIVE graph nodes

                stock.sample_graph = sample_graph
                stock.pct = torch.std(sample_graph[sym].x_samp) * 0.5 # try to predict a half std dev movement
                
                # Train model
                ################################
                latest = stock.live_price()[0][2] if live else sample_graph[sym].x_pred[0]

                y = movement(latest, pct=stock.pct.item())
                pred = self.model(sample_graph, pct=stock.pct)

                loss = self.criterion(pred, y)
                ix = step(sample_graph, ix, sym) if not live else None

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                ################################
                
                if self.buy_conditions(pred):
                    amount, used_ix = self.get_portion()
                    self.buy(sym, amount, used_ix) # if ix is None

            ### Inbetween period ######
            self.wait()
            self.update_graph()

            if self.trailing_prices is None:
                self.trailing_prices = {sym: (stock.sample_graph.buy_price, 0) for sym, stock in self.symbols.items()
                                            if stock is not None}
            ###########################
            
            # Checks for selling conditions in all symbols
            for sym in self.symbols:
                stock = self.stocks[sym]
                # Sell at profit or sell at the end of the prediction time
                if self.pull_back(sym) or not self.symbols[sym].sample_graph.x_pred: 
                    self.sell(sym) # sells all stocks owned of that symbol


    def sample_live(self, graph, sym): ###
        """ Sample live stock prices """
        self.update_graph()
        sample_graph, ix = sample(self.graph, sym, sample_len=self.sample_len, pred_len=self.pred_len, live=True)


    def update_graph(self): ###
        """ Update graph to append live stock prices """
        for sym in self.graph.metadata()[0]:
            stock = [s for s in self.stocks if s.sym==sym][0]
            x, t, node_ids = stock.live_price()

            ix, _ = self.graph[sym].edge_index[:, -1]
            edge_index = torch.tensor([[ix],
                                       [ix+1],])
            # x, t, node_ids, edge_index
            self.graph[sym].x = torch.cat([self.graph[sym].x, x])
            self.graph[sym].t = torch.cat([self.graph[sym].t, t])
            self.graph[sym].node_ids = torch.cat([self.graph[sym].node_ids, node_ids])
            self.graph[sym].edge_index = torch.cat([self.graph[sym].edge_index, edge_index])


    def get_portion(self):
        """
        Portions out the Account.net_value as follows:
            
            Tithe:
                10% - Always cash
            Investment Capital:
                90% - Divided into 8 portions of 11.25% each (90/8)        
        """
        inv_cap = self.net_value * 0.9 # Investment capital is 90%
        self.net_value -= inv_cap

        if self.portions is None:
            self.portions = torch.tensor([inv_cap / 8 for _ in range(8)], dtype=torch.float) # [1, 8]
            self.used = torch.zeros_like(self.portions)

        ix = (self.used==0).nonzero(as_tuple=True)[0][0] # first portion that is not 0 or negative, 
        amount = self.portions[ix] # select amount $

        self.portions[ix] -= amount # to 0
        self.used[ix] = 1 # mark used vector

        return amount, ix


    def buy(self, sym, amount, ix):
        """
        Buys stock of a symbol given the amount of $ provided by the ix'th portion.

        Updates:
            self.stocks
            self.current_bets
            self.history
            self.portions
        """
        if not self.stocks[sym]:
            stock = Stock(sym)
            self.stocks[sym] = stock
        else:
            stock = self.stocks[sym]

        stock.update()
        
        shares = int(amount / stock.price)
        self.current_bets[sym].append((shares * stock.price, shares, ix)) # sym: amount in stocks owned

        self.history.append((shares * stock.price, shares, ix, 'Buy'))

        left_over = amount - shares*stock.price
        self.portions[ix] += left_over


    def wait(self):
        """ Wait some time before continuing """
        time.sleep(10)


    def sell(self, sym):
        """
        Sells all stock portions of a given symbol at the current stock price
        and update portions accordingly.        

        Updates:
            self.used
            self.portions
            self.history
            self.current_bets
        """
        stock = self.stocks[sym]
        stock.update()

        for purchase_am, shares, ix in self.current_bets[sym]:
            self.used[ix] = 0 # set used vector space free
            self.portions[ix] += shares * stock.price

            self.history.append((shares * stock.price, shares, ix, 'Sell'))

        self.current_bets[sym] = list() # remove bet
        

    def buy_conditions(self, pred):
        """ Test if preditction is for upward movement """
        return pred==torch.tensor([1,0,0])



    def train(self):
        """
        Trains model to predict a one hot label of size [1, 3] 
        corresponding to the stock's movement given a x day interval.
        """
        for i in range(self.epochs):
            epoch_loss = 0
            for sym in self.symbols:
                print("\nTraining on: ", sym)

                for _ in range(1000): # sample 1000 times
                    sample_graph, ix = sample(self.graph, sym, sample_len=self.sample_len, pred_len=self.pred_len) # Samples Training graph nodes
                    pct = torch.std(sample_graph[sym].x_samp) * 0.5 # try to predict a half std dev movement
                    print("Percent pred: ", pct.item())

                    while sample_graph[sym].x_pred.numel() > 0: # while there is more data to predict

                        y_hat = self.model(sample_graph.x_samp_dict, sample_graph.edge_index_samp_dict) ###
                        y = movement(sample_graph[sym].x_pred[0], pct=pct.item())

                        loss = self.criterion(y_hat, y)
                        ix = step(sample_graph, ix, sym)

                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        self.optimizer.step()

                epoch_loss += loss.item()

                self.print(i, epoch_loss, freq=25)


    def pull_back(self, sym, pull=0.02):
        """
        Implements pull-back trading where you trail the price
        so it won't drop past a certain percent change since it's peak. 

        target_price [float]: peak price achieved 
        pull [float]: pull back percent from target_price
        """
        stock = self.stocks[sym]
        stock.update()
        self.stocks[sym] = stock

        trailing_price, active = self.trailing_prices[sym]
        target_price = stock.sample_graph.buy_price * (1+stock.pct)

        if trailing_price is None:
            trailing_price = target_price

        if stock.price >= target_price or active: 
            self.trailing_prices[sym][1] = True # set 'active' True

            if stock.price < trailing_price * (1-pull):
                self.trailing_prices[sym] = (None, 0) # reset symbol's trailing price once sold
                return True

            if stock.price() > trailing_price:
                self.trailing_prices[sym][0] = stock.price() # update 'trailing_price'

        return False


    def rank(self, ):
        """
        Ranks stock symbols by most to least profitable given a x day interval.
        
        The ranking occurs through simulated trading of the x day interval where 
        the 'model' buys a stock if it 
        """
        pass 


    def print(self, i, epoch_loss, freq=25):
        """ Prints epoch and loss data """
        if i % freq == 0:
            print(f"Epoch: {i+1}/{self.epochs}, Loss: {epoch_loss}")


# Suppose
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import random
import yfinance as yf

from torch_geometric.data import Data, Batch
from torch_geometric.data import TemporalData, HeteroData

from model import *
from utils import *


class Account:
    def __init__(self, sample_len=7*30, pred_len=7*5, hidden_dims=100, epochs=2000, learning_rate=1e-05, live=False):
        # Account info
        self.history = list() # Tape of transaction history
        self.net_value = 416.29
        self.portions = None # [1, 8]
        self.used = None # [1, 8]
        self.bets = 2 # Limited ammount of bets per day
        # Data
        self.symbols = ['SPY', 'TBLA', "MS", "PBM",] #"TCRX", "URG", "UROY", "UEC", "TBLA"]#"AMZN", "TSLA", "AAPL", "GOOGL", "META", "GM", "MS"]
        self.graph = build_graph(self.symbols, startDate, endDate)
        self.sample_graph = None
        self.sample_len, self.pred_len = sample_len, pred_len
        # Bets
        self.live = live
        self.current_bets = {sym: [] for sym in self.symbols if sym != 'SPY'} # sym: [(amount in $, shares), ..., ]
        # self.stocks = {sym: Stock(sym, live=live, graph=self.graph) for sym in self.symbols} # Graphs of stocks
        # Model
        self.model = MarketTransformer(metadata=self.graph.metadata())
        # Training
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        ##################################################


    def get_portion(self):
        """
        Portions out the Account.net_value as follows:

            Tithe:
                10% - Always cash
            Investment Capital:
                90% - Divided into 8 portions of 11.25% each (90/8)
        """

        if self.portions is None:
            inv_cap = self.net_value * 0.9 # Investment capital is 90%
            self.net_value -= inv_cap

            self.portions = torch.tensor([inv_cap / 8 for _ in range(8)], dtype=torch.float) # [1, 8]
            self.used = torch.zeros_like(self.portions)



        ixs = (self.used==0).nonzero(as_tuple=True)[0].tolist()# first portion that != 0 or negative,
        ix = random.choice(ixs)
        amount = torch.clone(self.portions[ix]) # select amount $

        self.portions[ix] -= amount.item() # to 0
        print("GET PORTION: ", self.portions, amount, ix)
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
        stock = self.stocks[sym]

        stock.update()
        price = stock.price

        if amount > price:
          shares = int(amount / price)
          left_over = (amount % price).item()
        else:
          print(sym, " stock price too high ", price, " for amount ", amount.item())
          self.portions[ix] += amount
          self.used[ix] = 0
          return

        # print(shares * stock.price, shares, ix)
        self.current_bets[sym].append((shares * price, shares, ix)) # sym: amount in stocks owned

        self.history.append((shares * price, shares, ix, 'Buy'))

        print("BUYING: ", sym, " for ", shares*price, " with ", left_over, " left over.")
        self.portions[ix] += left_over


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
            print("SOLD: ", sym, " for ", shares*stock.price, "*******************************************************************************************************************************************************************************************************************")

        self.current_bets[sym] = list() # remove bet


    def wait(self):
        """ Wait some time before continuing """
        time.sleep(10)


    def buy_conditions(self, pred):
        """ Test if preditction is for upward movement """
        return torch.equal(pred, torch.tensor([1,0,0]))


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


    def test(self): # Live trading simulation
        """
        Simulates the model predicting movement of the stocks given a sample interval
        and awaits for it to sell before beggining the process again.

        Goes in sequential order of a larger sample interval of a stock's history
        and gives a profitablility calculation based on model predictions.
        """
        while True:

          # Checks for buying conditions in all symbols
          for sym in [sym for sym, curr_bets in self.current_bets.items() if not curr_bets and sym != 'SPY']: #self.symbols:
              print("Checking for buying conditions: ", sym)

              stock = self.stocks[sym]
              stock.pct = torch.std(stock.sample_graph[sym].x_samp) * 0.5 # try to predict a half std dev movement

              # Train model
              ################################
              # latest = stock.live_price()[0][2] if self.live else sample_graph[sym].x_pred[0]

              y = torch.tensor([1, 0, 0]) #movement(latest, pct=stock.pct.item())
              pred = torch.clone(y) #self.model(sample_graph, pct=stock.pct)

              # loss = self.criterion(pred, y)

              # self.optimizer.zero_grad(set_to_none=True)
              # loss.backward()
              # self.optimizer.step()
              ################################

              if self.buy_conditions(pred):
                  if self.used != None and torch.all(self.used.bool()):
                    print("No more portions left.")
                    break

                  amount, used_ix = self.get_portion()
                  self.buy(sym, amount, used_ix) #if ix is None

          ### Inbetween period ######
          if self.live:
            self.wait()

          self.update_graph()
          ###########################

          # Checks for selling conditions in all symbols
          for sym in self.symbols:
              if sym == 'SPY':
                continue

              stock = self.stocks[sym]
              print("Monitoring: ", sym, "at ", stock.pct.item()*100, "%")

              # Sell at profit or sell at the end of the prediction time
              #print(stock.sample_graph[sym].x_pred, torch.any(stock.sample_graph[sym].x_pred))
              if self.pull_back(sym) or not torch.any(stock.sample_graph[sym].x_pred): # x_pred != empty
                  self.sell(sym) # sells all stocks owned of that symbol

                  if not torch.any(stock.sample_graph[sym].x_pred):
                    stock.sample_graph, stock.ix = sample(self.graph, sym, stock.sample_len, stock.pred_len, live=self.live)
                    self.stocks[sym] = stock # save new sample_graph


          self.print_acc()


    def print_acc(self):
        """ Print info for the account """

        if self.portions != None:
          net_val = self.net_value + torch.sum(self.portions) + sum([bet[0] for sym, bets in self.current_bets.items() for bet in bets])
        else:
          net_val = self.net_value

        for stock in self.stocks.values():
          if stock is not None:
              stock.update()

        print(
              f"""
                Account net_value: {net_val}\n

                Portions:
                {self.portions}\n
                Used:
                {self.used}\n

                Current bets: {self.current_bets}\n
                Trailing prices:\n
                { {sym: stock.trailing_data[0] for sym, stock in self.stocks.items() if stock != None and sym != 'SPY'} }

                LIVE PRICES:
                { {sym: stock.price for sym, stock in self.stocks.items() if stock is not None} }
                """
        )


    def update_graph(self): ###
        """
        Update graph to append live stock prices
        or step each stock's sample_graph
        """
        if self.live:
          for sym in self.graph.metadata()[0]:
            stock = self.stocks[sym]
            x, t, node_ids = stock.live_price()

            _, ix = self.graph[sym, 'next_in_sequence', sym].edge_index[:, -1]
            ix = ix.item()
            edge_index = torch.tensor([[ix],
                                       [ix+1],])

            # x, t, node_ids, edge_index
            self.graph[sym].x = torch.cat([self.graph[sym].x, x.unsqueeze(0)], dim=0)
            self.graph[sym].edge_index = torch.cat([self.graph[sym, 'next_in_sequence', sym].edge_index, edge_index], dim=1)
            # self.graph[sym].t = torch.cat([self.graph[sym].t, t])
            # self.graph[sym].node_ids = torch.cat([self.graph[sym].node_ids, node_ids])

        else:
          for sym in self.graph.metadata()[0]:
            stock = self.stocks[sym]

            if stock.ix < stock.sample_graph['SPY', 'same_time', sym].edge_index.shape[1]:
              stock.sample_graph, stock.ix = step(stock.sample_graph, stock.ix-1, sym)
              self.stocks[sym] = stock # update stocks

              stock.update()


    def pull_back(self, sym, pull=0.02):
        """
        Implements pull-back trading where you trail the price
        so it won't drop past a certain percent change since it's peak.

        target_price [float]: peak price achieved
        pull [float]: pull back percent from target_price
        """
        stock = self.stocks[sym]

        trailing_price, active = stock.trailing_data
        target_price = stock.sample_graph.buy_price * (1+stock.pct) # buy_price * 1.03

        if trailing_price is None:
            stock.trailing_data[0] = trailing_price = target_price.item()

        if stock.price >= target_price or active:
            #print(stock.trailing_data)
            stock.trailing_data[1] = 1 # set 'active' True

            if stock.price < trailing_price * (1-pull): # SELL CONDITION
                stock.trailing_data = [None, 0] # reset symbol's trailing price once sold
                return True

            if stock.price > trailing_price:
                #print(stock.trailing_data)
                stock.trailing_data[0] = stock.price # update 'trailing_price'

        return False


    def train(self):
        """
        Trains model to predict a one hot label of size [1, 3]
        corresponding to the stock's movement given a x day interval.
        """
        for i in range(self.epochs):
            epoch_loss = 0

            for _ in range(100): # sample 100 times
                print(self.symbols)
                sample_graph, curr_ixs = sample(self.graph, self.symbols, sample_len=self.sample_len, pred_len=self.pred_len, live=False) # Samples Training graph nodes
                curr_ixs = None

                while sample_graph['SPY'].x_pred.numel() > 0: # while there is more data to predict
                    y_hat_dict, time_dict_hat = self.model(sample_graph.x_samp_dict, sample_graph.edge_index_samp_dict) ###
                    y_dict, time_dict = movement(sample_graph[sym].x_pred[0], pct=pct.item())

                    # Calculate pct chg and time pred loss for each sym
                    for sym in sample_graph.node_types:
                        loss_y = self.criterion(y_hat_dict[sym], y_dict[sym])
                        loss_time = self.criterion(time_dict_hat[sym], time_dict[sym])
                        loss = loss_y + loss_time

                        epoch_loss += loss.item()

                        # Backward pass
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        self.optimizer.step()

                    curr_ixs = step(sample_graph, curr_ixs, sym)

            self.print(i, epoch_loss, freq=25)


class Stock:
    def __init__(self, sym, live, graph):
        # Metadata
        self.price = None # setted with live_price
        self.trailing_data = [None, 0]#, None] # trailing_price, active, used_ix
        self.live = live
        self.sym = sym
        # Graph
        self.sample_len = 7*30 # 7h * 30d
        self.pred_len = 7*5 # 7h * 5d
        self.sample_graph = None
        self.ix = None
        self.pct = None

        self.initialize_graph(graph)
        ########################

    def update(self):
        """
        Updates price, gets either the live price or the price
        at the current step of the sample_graph data object.
        """
        if self.live:
          price =  self.live_price()[0][2]
        else:
          #print(self.sample_graph[self.sym].x_pred)
          price = self.sample_graph[self.sym].x_samp[-1, 2].item() * self.sample_graph.buy_price
          print("Updated:", price, self.sym)

        self.price = price


    def live_price(self, interval='1m'):
        """ Gets info and sets live price of stock """
        s = yf.Ticker(self.sym)
        df = s.history(period='1d', interval=interval).iloc[-1]

        self.price = float(df['Close'].item()) # CLOSE price set as LIVE PRICE

        x =  torch.tensor(df[['High', 'Low', 'Close', 'Open']])
        t = None #df.index.strftime('%H%M%S').astype('int64')
        node_ids = None #torch.tensor(df.index.strftime('%Y%m%d%H%M%S').astype('int64'), dtype=torch.int64)

        return (x, t, node_ids)  # x,t,node_ids,edge_index


    def sample_live(self, graph, sym): ###
        """ Sample live stock prices """
        self.update() # sets live price

        sample_graph, ix = sample(graph, sym, self.sample_len, self.pred_len, live=self.live)
        return sample_graph, ix


    def initialize_graph(self, graph):
        """ Initialize sample_graph, ix, pct and price """
        if self.live:
            self.sample_graph, self.ix = self.sample_live(graph, self.sym)
        else:
            self.sample_graph, self.ix = sample(graph, self.sym, self.sample_len, self.pred_len, self.live)
            self.update()

        pct = None
        self.pct = pct

        self.trailing_data[0] = self.sample_graph.buy_price # missing used_ix

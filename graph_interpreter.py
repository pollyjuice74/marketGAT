# (x,y,z)->(time,stocks,price)

# prices of stock s on time t (s,t,p) will have a direction, since p is one dimensional
# representing the price at time t we can squeeze the dimentions and have a dataset that
# looks like (s,p) representing the normalized graph of prices of a stock s at a given
# interval of time [0,...,T].

# it can be represented as a Markov Noising Proceess: q(x_t0) = f(x_t0) + N
# as the stock s progresses thorugh time t at the current price p

# get the x_hat and z_hat estimate of the direction function x and the noise z, it is separated 
# from each of the stocks, all stocks can be cross analyzed and compared to eachother
# based on their extracted x_hat, and z_hat to create a general market multidim vector 
# of how a stock's x value is pointing in multidimensional space, each stock can have
# this representation, noise extreacted z could be cumulated and used to calculate
# how the market affects the price fluctuation of the individual stock.

class MarketAverage

class StockAttn

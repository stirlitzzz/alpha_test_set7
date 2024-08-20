import numpy as np
import pandas as pd



def main():
    dates=pd.date_range(start="2021-01-01",end="2021-01-10")
    ticker=["AAPL","MSFT","GOOGL","AMZN","TSLA","FB","NVDA","PYPL","INTC","CSCO"]
    #create a random np array of weights with dimensions [len(dates),len(ticker)]
    #normalize
    #np_weights=np_weights/np_weights.sum(axis=1)[:,None]
    #print(f'np_weights.sum(axis=1)={np_weights.sum(axis=1)}')


    #instrumentxdate array
    #np_prices: instrument prices
    #np_returns: instrument returns
    #weighted_returns: weighted returns
    #np_weights: weights (normalized)

    #instrument array
    #np_vols: instrument vols

    #date array     
    #leverage: leverage
    #port_returns: portfolio returns
    #levered_returns: levered returns
    #cash: cash


    #genarage random prices and returns
    np_vols=np.random.uniform(0.15,.45,(1,len(ticker)))
    np_prices=np.zeros((len(dates),len(ticker)))
    np_prices[0,:]=100

    np_returns=np.random.randn(len(dates),len(ticker))
    np_returns=np_returns*np_vols/np.sqrt(252)
    np_prices[1:,:]=np_prices[0,:]*(1+np_returns[1:,:])
    #could also generate with log returns, then recalculate percent returns and use them for
    #to portfolio returns
    print(f'np_prices={np_prices}')



    np_weights=np.random.rand(len(dates),len(ticker))
    np_weights=np_weights/np_weights.sum(axis=1).reshape(-1,1)

    leverage=np.random.uniform(1,2,(len(dates),1))

    #log_rets=np.log(1+np_returns)
    weighted_returns=(np_weights[:-1,:]*np_returns[1:,:])
    port_returns=weighted_returns.sum(axis=1).reshape(-1,1)
    levered_returns=port_returns*leverage[:-1,:]
    print(f'port_returns.shape={port_returns.shape}')
    print(f'levered_returns.shape={levered_returns.shape}')
    initial_cash=100
    cash=initial_cash*(1+levered_returns).cumprod()
    print(f'levered_returns={levered_returns}')
    print(f'cash={cash}')
    #port_returns=(np_weights[:-1,:]*log_rets[1:,:])


    #port_prices=np.exp(np.cumsum(port_returns,axis=0))
    #print(f'port_prices={port_prices}')
    #print(f'port_returns={port_returns}')

    #input(f'np_prices={np_prices}')
    #print(f'np_weights.sum(axis=1)={np_weights.sum(axis=1)}')
    #print(dates)
    #print(np_weights)
    #print(np_weights[0,:].sum())

if __name__=="__main__":
    main()


import numpy as np
import pandas as pd
from alpha_output import output_dict_as_xlsx

def generate_market_data(start_date="2021-01-01", end_date="2021-01-15",num_tickers=5):
    dates=pd.date_range(start=start_date,end=end_date)
    tickers=[f"ticker{i}" for i in range(num_tickers)]

    np_vols=np.random.uniform(0.15,.45,(1,len(tickers)))
    np_prices=np.zeros((len(dates),len(tickers)))
    np_prices[0,:]=100
    np_returns=np.random.randn(len(dates),len(tickers))
    np_returns=np_returns*np_vols/np.sqrt(252)
    np_prices[1:,:]=np_prices[0,:]*(1+np_returns[1:,:])
    return(dates,tickers,np_prices,np_returns,np_vols)

def generate_weights(dates,tickers):
    np_weights=np.random.rand(len(dates),len(tickers))
    np_weights=np_weights/np_weights.sum(axis=1).reshape(-1,1)
    leverage=np.random.uniform(1,2,(len(dates),1))

    return (np_weights, leverage)

def output_dict_as_xlsx(dict_data, path):
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        for sheet_name, df in dict_data.items():
            #if (df.columns[0]=="datetime"):
            #    df=YahooDataConverter.make_timezone_naive(df)
            print(f'sheet_name: {sheet_name}')
            print(f'df.index: {df.index}')
            print(f'type(df.index): {type(df.index)}')
            if(type(df.index)==pd.DatetimeIndex):
                df.index=df.index.tz_localize(None)
                df.index.name='datetime'
            if("datetime" in df.columns):
                df["datetime"]=df["datetime"].dt.tz_localize(None)
            df.to_excel(writer, sheet_name=sheet_name, index=True)  


def generate_random_set(start_date="2021-01-01", end_date="2021-01-15",num_tickers=5):
    dates=pd.date_range(start=start_date,end=end_date)
    tickers=[f"ticker{i}" for i in range(num_tickers)]
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
    np_vols=np.random.uniform(0.15,.45,(1,len(tickers)))
    np_prices=np.zeros((len(dates),len(tickers)))
    np_prices[0,:]=100

    np_returns=np.random.randn(len(dates),len(tickers))
    np_returns[0,:]=0
    np_returns=np_returns*np_vols/np.sqrt(252)
    np_prices[1:,:]=np_prices[0,:]*(1+np_returns[1:,:])

    print(f'np_prices={np_prices}')
    return(dates,tickers,np_prices,np_returns,np_vols)

def portfolio_pnl(np_weights,np_returns,leverage):
    """
    np_weights: weights (normalized)
    np_returns: instrument returns
    leverage: leverage

    port_returns: portfolio returns
    levered_returns: levered returns
    cash: cash

    port_returns->sum_{i}(weights_{t-1}*returns_{t})
    levered_returns->port_returns*leverage_{t-1}
    """
    weighted_returns=(np_weights[:-1,:]*np_returns[1:,:])

    port_returns=weighted_returns.sum(axis=1).reshape(-1,1)
    levered_returns=port_returns*leverage[:-1,:]
    print(f'port_returns.shape={port_returns.shape}')
    print(f'levered_returns.shape={levered_returns.shape}')
    initial_cash=100
    cash=initial_cash*(1+levered_returns).cumprod()
    cash=np.concatenate(([initial_cash],cash.flatten())).reshape(-1,1)

    return (port_returns,levered_returns,cash)

def portfolio_one_day(np_weights,np_returns,leverage):
    return (np_weights.T.dot(np_returns.T)*leverage)


def create_market_data(tickers,ticker_dfs):
    def is_any_one(x):
        return int(np.any(x))
    dfs={} 
    closes, eligibles, vols, rets = [], [], [], []
    trade_range = pd.date_range(start=ticker_dfs[tickers[0]].index[0],end=ticker_dfs[tickers[0]].index[-1])
    df=pd.DataFrame(index=trade_range)
    for inst in tickers:
        inst_vol = (-1 + ticker_dfs[inst]["close"]/ticker_dfs[inst]["close"].shift(1)).rolling(30).std()
        dfs = {}

        dfs = df.join(ticker_dfs[inst]["close"]).fillna(method="ffill").fillna(method="bfill")
        dfs["ret"] = -1 + dfs["close"]/dfs["close"].shift(1)
        dfs["vol"] = inst_vol
        dfs["vol"] = dfs["vol"].fillna(method="ffill").fillna(0)       
        dfs["vol"] = np.where(dfs["vol"] < 0.005, 0.005, dfs["vol"])

        sampled = (dfs["close"] != dfs["close"].shift(1).fillna(method="bfill"))
        eligible = sampled.rolling(5).apply(is_any_one,raw=True).fillna(0)
        eligibles.append(eligible.astype(int) & (dfs["close"] > 0).astype(int))
        closes.append(dfs["close"])
        vols.append(dfs["vol"])
        print(f'dfs["vol"]={dfs["vol"]}')
        rets.append(dfs["ret"])



    eligiblesdf = pd.concat(eligibles,axis=1)
    eligiblesdf.columns = tickers
    closedf = pd.concat(closes,axis=1)
    closedf.columns = tickers
    print(f'vols={vols}')
    voldf = pd.concat(vols,axis=1)
    voldf.columns = tickers
    retdf = pd.concat(rets,axis=1)
    retdf.columns = tickers
    result={"closes":closedf,"eligibles":eligiblesdf,"vols":voldf,"rets":retdf}

    return result

def pre_compute(tickers,ticker_dfs, start_date="2021-01-01", end_date="2021-01-15"):
    dfs={}
    dfs_out={}
    fields=["open","high","low","close","volume"]
    df=pd.DataFrame(index=pd.date_range(start=start_date,end=end_date))

    for field in fields:
        dfs[field]=[]

    for inst in tickers:
        df_resample=df.join(ticker_dfs[inst]).fillna(method="ffill").fillna(method="bfill")
        for field in fields:
            dfs[field].append(df_resample[field])
    for field in fields:
        dfs_out[field]=pd.concat(dfs[field],axis=1)
        dfs_out[field].columns=tickers
    return dfs_out

def alpha1_compute(tickers,dfs):
    op1=dfs["volume"]
    op2=(dfs["close"]-dfs["low"])-(dfs["high"]-dfs["close"])
    op3=dfs["high"]-dfs["low"]
    op4=op1*op2/op3
    dfs=deepcopy(dfs)
    dfs["op4"]=op4

    zscore = lambda x: (x - np.nanmean(x))/np.nanstd(x)
    dfs["czscore"] = op4.fillna(method="ffill").apply(zscore, axis=1, raw=True)
    dfs["alpha"] = dfs["czscore"].rolling(12,axis=0).mean() * -1
    input(f'dfs.keys={dfs.keys()}')
    dfs["eligibles"]=dfs["eligibles"] & (~pd.isna(dfs["alpha"]))
    masked_df = dfs["alpha"]/dfs["eligibles"]
    masked_df = masked_df.replace([-np.inf, np.inf], np.nan)
    num_eligibles = dfs["eligibles"].sum(axis=1)
    rankdf= masked_df.rank(axis=1,method="average",na_option="keep",ascending=True)
    shortdf = rankdf.apply(lambda col: col <= num_eligibles.values/4, axis=0,raw=True)
    longdf = rankdf.apply(lambda col: col > np.ceil(num_eligibles - num_eligibles/4), axis=0, raw=True)
   
    forecast_df = -1*shortdf.astype(np.int32) + longdf.astype(np.int32)
    dfs["forecast"]= forecast_df
    return dfs

def compute_signal_distribution(dfs):
    forecasts=dfs["forecast"].values
    input(f'forecasts={forecasts}')
    return forecasts

def compute_weights(dfs,forecast):
    if (type(forecast)==pd.DataFrame):
        forecast=forecast.values
    forecasts = forecast/dfs["eligibles"].values
    forecasts = np.nan_to_num(forecasts,nan=0,posinf=0,neginf=0)
    foreacast_chips=np.sum(np.abs(forecast),axis=1)
    weights=forecast/foreacast_chips[:,None]
    input(f'weights={weights}')
    dfs["weights"]=pd.DataFrame(weights,index=dfs["forecast"].index,columns=dfs["forecast"].columns)
    return weights


    """
            inst_df = self.dfs[inst]
            op1 = inst_df.volume
            op2 = (inst_df.close - inst_df.low) - (inst_df.high - inst_df.close)
            op3 = inst_df.high - inst_df.low
            op4 = op1 * op2 / op3       
            self.op4s[inst] = op4
    """
from copy import deepcopy
def default_compute(dfs):
    df=deepcopy(dfs)
    def is_any_one(x):
        return int(np.any(x))
    df["ret"] = -1 + df["close"]/df["close"].shift(1)
    df["vol"] = df["ret"].rolling(30).std()
    df["vol"] = df["vol"].fillna(method="ffill").fillna(0)       
    vol = np.where(df["vol"] < 0.005, 0.005, df["vol"])
    df["vol"]=pd.DataFrame(vol,index=df["close"].index,columns=df["close"].columns)

    sampled = (dfs["close"] != dfs["close"].shift(1).fillna(method="bfill"))
    eligible = sampled.rolling(5,axis=0).apply(is_any_one,raw=True).fillna(0)
    eligible = eligible.astype(int) & (df["close"] > 0).astype(int)
    df["eligibles"]=eligible
    return df





    





def main():
    (dates,ticker,np_prices,np_returns,np_vols)=generate_random_set(start_date="2021-01-01", end_date="2021-01-15",num_tickers=5)
    (np_weights,leverage)=generate_weights(dates,ticker)
    
    (port_returns,levered_returns,cash)=portfolio_pnl(np_weights,np_returns,leverage)
    port_returns=np.concatenate(([0],port_returns.flatten())).reshape(-1,1)
    levered_returns=np.concatenate(([0],levered_returns.flatten())).reshape(-1,1)
    units=cash*np_weights*leverage/np_prices

    tables={"prices":np_prices,
            "returns":np_returns,
            "weights":np_weights,
            "units":units}
    port_tables={"port_returns":port_returns,
            "levered_returns":levered_returns,
            "cash":cash,
            "leverage":leverage}

    for key,table in tables.items():
        table_df=pd.DataFrame(table,index=dates,columns=ticker)
        tables[key]=table_df

    for key,table in port_tables.items():
        print(f'key={key}')
        table_df=pd.DataFrame(table,index=dates,columns=["port"])
        port_tables[key]=table_df

    output_dict_as_xlsx({**tables, **port_tables},"tables.xlsx")
    #output_dict_as_xlsx(port_tables,"port_tables.xlsx")
    one_day_pnl=portfolio_one_day(np_weights[0],np_returns[1],leverage[0])
    print(f'one_day_pnl={one_day_pnl}')

    

    def daily_pnl_input_generator():
        for i in range(1,len(dates)):
            yield (np_weights[i-1],np_returns[i],leverage[i-1])

    daily_pnl=[portfolio_one_day(*args) for args in daily_pnl_input_generator()]
    daily_pnl=np.array(daily_pnl)
    print(f'daily_pnl={daily_pnl}')

    print(f'levered_returns={levered_returns}')
    print(f'cash={cash}')
    units=cash*np_weights*leverage/np_prices
    print(f'units={units}')

if __name__=="__main__":
    main()


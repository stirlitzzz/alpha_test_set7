import pytz
import yfinance
import requests
import threading
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from utils import timeme
from utils import save_pickle, load_pickle
from alpha1 import Alpha1
from alpha2 import Alpha2
from alpha3 import Alpha3
from utils import Portfolio

def get_sp500_tickers():
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content,'html')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table))
    tickers = list(df[0].Symbol)
    return tickers

def get_history(ticker, period_start, period_end, granularity="1d", tries=0):
    try:
        df = yfinance.Ticker(ticker).history(
            start=period_start,
            end=period_end,
            interval=granularity,
            auto_adjust=True
        ).reset_index()
    except Exception as err:
        if tries < 5:
            return get_history(ticker, period_start, period_end, granularity, tries+1)
        return pd.DataFrame()
    
    df = df.rename(columns={
        "Date":"datetime",
        "Open":"open",
        "High":"high",
        "Low":"low",
        "Close":"close",
        "Volume":"volume"
    })
    if df.empty:
        return pd.DataFrame()
    
    df["datetime"] = df["datetime"].dt.tz_localize(pytz.utc)
    df = df.drop(columns=["Dividends", "Stock Splits"])
    df = df.set_index("datetime",drop=True)
    return df

def get_histories(tickers, period_starts,period_ends, granularity="1d"):
    dfs = [None]*len(tickers)
    def _helper(i):
        print(tickers[i])
        df = get_history(
            tickers[i],
            period_starts[i], 
            period_ends[i], 
            granularity=granularity
        )
        dfs[i] = df
    threads = [threading.Thread(target=_helper,args=(i,)) for i in range(len(tickers))]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    tickers = [tickers[i] for i in range(len(tickers)) if not dfs[i].empty]
    dfs = [df for df in dfs if not df.empty]
    return tickers, dfs

def get_ticker_dfs(start,end):
    from utils import load_pickle,save_pickle
    try:
        tickers, ticker_dfs = load_pickle("dataset.obj")
    except Exception as err:
        tickers = get_sp500_tickers()
        starts=[start]*len(tickers)
        ends=[end]*len(tickers)
        tickers,dfs = get_histories(tickers,starts,ends,granularity="1d")
        ticker_dfs = {ticker:df for ticker,df in zip(tickers,dfs)}
        save_pickle("dataset.obj", (tickers,ticker_dfs))
    return tickers, ticker_dfs 

def main():
    period_start = datetime(2010,1,1, tzinfo=pytz.utc)
    period_end = datetime.now(pytz.utc)
    tickers, ticker_dfs = get_ticker_dfs(start=period_start,end=period_end)
    testfor = 200
    print(f"testing {testfor} out of {len(tickers)} tickers")
    tickers = tickers[:testfor]

    alpha1 = Alpha1(insts=tickers,dfs=ticker_dfs,start=period_start,end=period_end)
    alpha2 = Alpha2(insts=tickers,dfs=ticker_dfs,start=period_start,end=period_end)
    alpha3 = Alpha3(insts=tickers,dfs=ticker_dfs,start=period_start,end=period_end)

    df1 = alpha1.run_simulation()
    df2 = alpha2.run_simulation()
    df3 = alpha3.run_simulation()
    
    print(list(df1.capital)[-1])
    print(list(df2.capital)[-1])
    print(list(df3.capital)[-1])

if __name__ == "__main__":
    main()

'''
@timeme: run_simulation took 362.73129987716675 seconds
62943.04905430462
@timeme: run_simulation took 393.18128275871277 seconds
26599.72491243218
@timeme: run_simulation took 390.20913791656494 seconds
113583.83554499355

@timeme: run_simulation took 12.972365140914917 seconds
62943.04905430462
@timeme: run_simulation took 11.576030254364014 seconds
26599.724912432248
@timeme: run_simulation took 14.0901460647583 seconds
113583.8355449932 

@timeme: run_simulation took 12.089932918548584 seconds
@timeme: run_simulation took 10.329317092895508 seconds
@timeme: run_simulation took 10.602593898773193 seconds
'''
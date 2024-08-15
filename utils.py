import lzma
import dill as pickle

import time
from functools import wraps
def timeme(func):
    @wraps(func)
    def timediff(*args,**kwargs):
        a = time.time()
        result = func(*args,**kwargs)
        b = time.time()
        print(f"@timeme: {func.__name__} took {b - a} seconds")
        return result
    return timediff

def load_pickle(path):
    with lzma.open(path,"rb") as fp:
        file = pickle.load(fp)
    return file

def save_pickle(path,obj):
    with lzma.open(path,"wb") as fp:
        pickle.dump(obj,fp)

def get_pnl_stats(last_weights, last_units, prev_close, ret_row, leverages):
    ret_row = np.nan_to_num(ret_row,nan=0,posinf=0,neginf=0)
    day_pnl = np.sum(last_units * prev_close * ret_row)
    nominal_ret = np.dot(last_weights, ret_row)
    capital_ret = nominal_ret * leverages[-1]
    return day_pnl, nominal_ret, capital_ret   

import numpy as np
import pandas as pd
from copy import deepcopy

class AbstractImplementationException(Exception):
    pass

class Alpha():

    def __init__(self, insts, dfs, start, end, portfolio_vol=0.20):
        self.insts = insts
        self.dfs = deepcopy(dfs)
        self.start = start 
        self.end = end
        self.portfolio_vol = portfolio_vol
    
    def pre_compute(self,trade_range):
        pass

    def post_compute(self,trade_range):
        pass

    def compute_signal_distribution(self, eligibles, date):
        raise AbstractImplementationException("no concrete implementation for signal generation")

    def get_strat_scaler(self, target_vol, ewmas, ewstrats):
        ann_realized_vol = np.sqrt(ewmas[-1] * 253)
        return target_vol / ann_realized_vol * ewstrats[-1]

    def compute_meta_info(self,trade_range):
        self.pre_compute(trade_range=trade_range)
        
        def is_any_one(x):
            return int(np.any(x))
        
        closes, eligibles, vols, rets = [], [], [], []
        for inst in self.insts:
            df=pd.DataFrame(index=trade_range)
            inst_vol = (-1 + self.dfs[inst]["close"]/self.dfs[inst]["close"].shift(1)).rolling(30).std()
            self.dfs[inst] = df.join(self.dfs[inst]).fillna(method="ffill").fillna(method="bfill")
            self.dfs[inst]["ret"] = -1 + self.dfs[inst]["close"]/self.dfs[inst]["close"].shift(1)
            self.dfs[inst]["vol"] = inst_vol
            self.dfs[inst]["vol"] = self.dfs[inst]["vol"].fillna(method="ffill").fillna(0)       
            self.dfs[inst]["vol"] = np.where(self.dfs[inst]["vol"] < 0.005, 0.005, self.dfs[inst]["vol"])
            sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).fillna(method="bfill")
            eligible = sampled.rolling(5).apply(is_any_one,raw=True).fillna(0)
            eligibles.append(eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int))
            closes.append(self.dfs[inst]["close"])
            vols.append(self.dfs[inst]["vol"])
            rets.append(self.dfs[inst]["ret"])

        self.eligiblesdf = pd.concat(eligibles,axis=1)
        self.eligiblesdf.columns = self.insts
        self.closedf = pd.concat(closes,axis=1)
        self.closedf.columns = self.insts
        self.voldf = pd.concat(vols,axis=1)
        self.voldf.columns = self.insts
        self.retdf = pd.concat(rets,axis=1)
        self.retdf.columns = self.insts

        self.post_compute(trade_range=trade_range)
        return

    @timeme
    def run_simulation(self):
        date_range = pd.date_range(start=self.start,end=self.end, freq="D")
        self.compute_meta_info(trade_range=date_range)
        units_held, weights_held = [],[]
        close_prev = None
        ewmas, ewstrats = [0.01], [1]
        strat_scalars = []
        capitals, nominal_rets, capital_rets = [10000.0],[0.0],[0.0]
        nominals, leverages = [],[]
        for data in self.zip_data_generator():
            portfolio_i=data["portfolio_i"]
            ret_i = data["ret_i"]
            ret_row = data["ret_row"]
            close_row = data["close_row"]
            eligibles_row = data["eligibles_row"]
            vol_row = data["vol_row"]
            strat_scalar = 2
           
            if portfolio_i != 0:
                strat_scalar = self.get_strat_scaler(
                    target_vol=self.portfolio_vol,
                    ewmas=ewmas,
                    ewstrats=ewstrats
                )

                day_pnl, nominal_ret, capital_ret = get_pnl_stats(
                    last_weights=weights_held[-1], 
                    last_units=units_held[-1], 
                    prev_close=close_prev, 
                    ret_row=ret_row, 
                    leverages=leverages
                )
                
                capitals.append(capitals[-1] + day_pnl)
                nominal_rets.append(nominal_ret)
                capital_rets.append(capital_ret)
                ewmas.append(0.06 * (capital_ret**2) + 0.94 * ewmas[-1] if capital_ret != 0 else ewmas[-1])
                ewstrats.append(0.06 * strat_scalar + 0.94 * ewstrats[-1] if capital_ret != 0 else ewstrats[-1])

            strat_scalars.append(strat_scalar)
            forecasts = self.compute_signal_distribution(
                eligibles_row,
                ret_i
            )
            if type(forecasts) == pd.Series: forecasts = forecasts.values
            forecasts = forecasts/eligibles_row
            forecasts = np.nan_to_num(forecasts,nan=0,posinf=0,neginf=0)
            forecast_chips = np.sum(np.abs(forecasts))
            vol_target = (self.portfolio_vol / np.sqrt(253)) \
                * capitals[-1]
            positions = strat_scalar * \
                    forecasts / forecast_chips  \
                    * vol_target \
                    / (vol_row * close_row) if forecast_chips != 0 else np.zeros(len(self.insts))
        
            positions = np.nan_to_num(positions,nan=0,posinf=0,neginf=0)
            nominal_tot = np.linalg.norm(positions * close_row, ord=1)
            units_held.append(positions)
            weights = positions * close_row / nominal_tot
            weights = np.nan_to_num(weights,nan=0,posinf=0,neginf=0)
            weights_held.append(weights)

            nominals.append(nominal_tot)
            leverages.append(nominal_tot/capitals[-1])
            close_prev = close_row
        
        units_df = pd.DataFrame(data=units_held, index=date_range, columns=[inst + " units" for inst in self.insts])
        weights_df = pd.DataFrame(data=weights_held, index=date_range, columns=[inst + " w" for inst in self.insts])
        nom_ser = pd.Series(data=nominals, index=date_range, name="nominal_tot")
        lev_ser = pd.Series(data=leverages, index=date_range, name="leverages")
        cap_ser = pd.Series(data=capitals, index=date_range, name="capital")
        nomret_ser = pd.Series(data=nominal_rets, index=date_range, name="nominal_ret")
        capret_ser = pd.Series(data=capital_rets, index=date_range, name="capital_ret")
        scaler__ser = pd.Series(data=strat_scalars, index=date_range, name="strat_scalar")
        portfolio_df = pd.concat([
            units_df,
            weights_df,\
            lev_ser,
            scaler__ser,
            nom_ser,
            nomret_ser,
            capret_ser,
            cap_ser
        ],axis=1)
        return portfolio_df

    def zip_data_generator(self):
        for (portfolio_i),\
            (ret_i, ret_row), \
            (close_i, close_row), \
            (eligibles_i, eligibles_row), \
            (vol_i, vol_row) in zip(
                range(len(self.retdf)),
                self.retdf.iterrows(),
                self.closedf.iterrows(),
                self.eligiblesdf.iterrows(),
                self.voldf.iterrows()
            ):
            yield {
                "portfolio_i": portfolio_i,
                "ret_i": ret_i,
                "ret_row": ret_row.values,
                "close_row": close_row.values,
                "eligibles_row": eligibles_row.values,
                "vol_row": vol_row.values,
            }

from collections import defaultdict
class Portfolio(Alpha):
    
    def __init__(self,insts,dfs,start,end,stratdfs):
        super().__init__(insts,dfs,start,end)
        self.stratdfs=stratdfs

    def post_compute(self,trade_range):
        self.positions = {}
        for inst in self.insts:
            inst_weights = pd.DataFrame(index=trade_range)
            for i in range(len(self.stratdfs)):
                inst_weights[i] = self.stratdfs[i]["{} w".format(inst)]\
                    * self.stratdfs[i]["leverage"]
                inst_weights[i] = inst_weights[i].fillna(method="ffill").fillna(0.0)
            self.positions[inst] = inst_weights

    def compute_signal_distribution(self, eligibles, date):
        forecasts = defaultdict(float)
        for inst in self.insts:
            for i in range(len(self.stratdfs)):
                forecasts[inst] += self.positions[inst].at[date, i] * (1/len(self.stratdfs))
                #parity risk allocation
        return forecasts, np.sum(np.abs(list(forecasts.values())))


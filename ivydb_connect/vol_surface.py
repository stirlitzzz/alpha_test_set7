import numpy as np
from scipy.stats import linregress


def spot_vol(atm_vol,slope, expiry, strike, anchor_strike=1.0):
    result=atm_vol+slope*10/np.sqrt(expiry)*np.log(strike/anchor_strike)
    return result

def fit_vol_surface(atm_vol, expiry, strike, vols, anchor_strike=1.0):
    x=np.log(strike/anchor_strike)
    y=vols-atm_vol
    #slope, intercept, r_value, p_value, std_err=linregress(x,y)
    slope=np.polyfit(x,y,1)
    slope=slope/10*np.sqrt(expiry)
    return atm_vol, slope
    

atm_vol=0.2
slope=-.025
expiry=1.0
strike=np.array([0.9,1.0,1.1])
anchor_strike=1.0
spot_vols=spot_vol(atm_vol,slope, expiry, strike, anchor_strike)

atm_vol2, slope2=fit_vol_surface(atm_vol, expiry, strike, spot_vols, anchor_strike)

print(f'Atm vol: {atm_vol}, slope: {slope}, expiry: {expiry}, strike: {strike}, anchor_strike: {anchor_strike}')
print(f'Spot vols: {spot_vols}')
print(f'Atm vol2: {atm_vol2}, slope2: {slope2}')

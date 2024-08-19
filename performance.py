import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def performance_measures(r, plot=False, path="/images"):
    moment = lambda x,k: np.mean((x-np.mean(x))**k)
    stdmoment = lambda x,k: moment(x,k)/moment(x,2)**(k/2)
    cr = np.cumprod(1 + r)
    lr = np.log(cr)
    mdd=cr/cr.cummax() - 1
    rdd_fn = lambda cr,pr: cr/cr.rolling(pr).max() - 1
    rmdd_fn = lambda cr,pr: rdd_fn(cr,pr).rolling(pr).min()
    srtno = np.mean(r.values)/np.std(r.values[r.values<0])*np.sqrt(253)
    shrpe = np.mean(r.values)/np.std(r.values)*np.sqrt(253)
    mu1 = np.mean(r)*253
    med = np.median(r)*253
    stdev = np.std(r)*np.sqrt(253)
    var = stdev**2
    skw = stdmoment(r,3)
    exkurt = stdmoment(r,4)-3
    cagr_fn = lambda cr: (cr[-1]/cr[0])**(1/len(cr))-1
    cagr_ann_fn = lambda cr: ((1+cagr_fn(cr))**253) - 1
    cagr = cagr_ann_fn(cr)
    rcagr = cr.rolling(5*253).apply(cagr_ann_fn,raw=True)
    calmar = cr.rolling(3*253).apply(cagr_ann_fn,raw=True) / rmdd_fn(cr=cr,pr=3*253)*-1
    var95 = np.percentile(r,0.95)
    cvar = r[r < var95].mean()
    if plot:
        import os
        from pathlib import Path
        Path(os.path.abspath(os.getcwd()+path)).mkdir(parents=True,exist_ok=True)
        ax=sns.histplot(r,stat="probability")
        ax.axvline(x=np.mean(r), linestyle="-")
        ax.axvline(x=np.median(r), linestyle="dotted")
        plt.savefig(f".{path}/kde.png")
        plt.close()

        sns.lineplot(lr)
        plt.savefig(f".{path}/log_ret.png")
        plt.close()

        sns.lineplot(rdd_fn(cr,253))
        sns.lineplot(rmdd_fn(cr,253))
        plt.savefig(f".{path}/drawdowns.png")
        plt.close()

    return {
        "cum_ret": cr,
        "log_ret": lr,
        "max_dd": mdd,
        "cagr": cagr,
        "srtno": srtno,
        "sharpe": shrpe,
        "mean_ret": mu1,
        "median_ret": med,
        "vol": stdev,
        "var": var,
        "skew": skw,
        "exkurt": exkurt,
        "cagr": cagr,
        "rcagr": rcagr,
        "calmar": calmar,
        "var95": var95,
        "cvar": cvar
    }
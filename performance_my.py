import numpy as np

class PerformanceMeasure():
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.0):
        return np.mean(returns - risk_free_rate) / np.std(returns)*np.sqrt(252)

    @staticmethod
    def sortino_ratio(returns, risk_free_rate=0.0):
        return np.mean(returns - risk_free_rate) / np.std(returns[returns < 0])*np.sqrt(252)

    @staticmethod
    def max_drawdown(returns):
        cum_returns = np.cumprod(1 + returns)
        return np.min(cum_returns/np.maximum.accumulate(cum_returns)-1)

    @staticmethod
    def calmar_ratio(returns, risk_free_rate=0.0):
        return np.mean(returns - risk_free_rate) / PerformanceMeasure.max_drawdown(returns)

    @staticmethod
    def annualized_volatility(returns):
        return np.std(returns) * np.sqrt(252)

    @staticmethod
    def annualized_return(returns):
        return np.mean(returns) * 252

    @staticmethod
    def cagr(returns):
        return (returns[-1] / returns[0]) ** (1 / len(returns)) - 1

    @staticmethod
    def omega_ratio(returns, risk_free_rate=0.0):
        return np.mean(returns - risk_free_rate) / np.percentile(returns, 5)

    @staticmethod
    def compute_all_metrics(cap_returns):
        return {
            "sharpe_ratio": PerformanceMeasure.sharpe_ratio(cap_returns),
            "sortino_ratio": PerformanceMeasure.sortino_ratio(cap_returns),
            "max_drawdown": PerformanceMeasure.max_drawdown(cap_returns),
            "calmar_ratio": PerformanceMeasure.calmar_ratio(cap_returns),
            "annualized_volatility": PerformanceMeasure.annualized_volatility(cap_returns),
            "annualized_return": PerformanceMeasure.annualized_return(cap_returns),
            "cagr": PerformanceMeasure.cagr(cap_returns),
            "omega_ratio": PerformanceMeasure.omega_ratio(cap_returns)
        }
        

import numpy as np
from scipy.stats import norm

def BlackScholes(call:bool, S0:float, K:float, r:float, vol:float, T:float):
    d1 = (np.log(S0 / K) + (r + (vol**2) / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    if call:
        option_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1

    return option_price, delta
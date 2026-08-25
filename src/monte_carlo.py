import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from gbm import simulate_gbm, simulate_gbm_cont, simulate_gbm_antithetic

def MC_value_option(call:bool, n_paths:int, S0:float, K:float, r:float, vol:float, T:float, alpha:float=0.05, fixed_Z=None):
    terminal_prices = simulate_gbm_cont(
        n_paths=n_paths,
        drift=r,
        vol=vol,
        S0=S0,
        T=T,
        fixed_Z=fixed_Z
    )

    if call:
        payoffs = [max(terminal_price - K, 0) for terminal_price in terminal_prices]
    else:
        payoffs = [max(K - terminal_price, 0) for terminal_price in terminal_prices]

    discounted_payoffs = [np.exp(-r * T) * p for p in payoffs]

    value_est = np.mean(discounted_payoffs)

    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)

    confidence_bounds = [value_est - norm.ppf(1 - (alpha / 2)) * std_error, value_est + norm.ppf(1 - (alpha / 2)) * std_error]

    return value_est, confidence_bounds

def MC_value_option_antithetic(call:bool, n_paths:int, S0:float, K:float, r:float, vol:float, T:float, alpha:float=0.05):
    terminal_prices = simulate_gbm_antithetic(
        n_paths=n_paths,
        drift=r,
        vol=vol,
        S0=S0,
        T=T,
    )

    if call:
        payoffs = [max(terminal_price - K, 0) for terminal_price in terminal_prices]
    else:
        payoffs = [max(K - terminal_price, 0) for terminal_price in terminal_prices]

    discounted_payoffs = np.exp(-r * T) * np.array(payoffs)

    pair_payoffs = (discounted_payoffs[0::2] + discounted_payoffs[1::2]) / 2

    value_est = np.mean(pair_payoffs)

    std_error = (np.std(pair_payoffs) / np.sqrt(n_paths / 2))

    confidence_bounds = [value_est - norm.ppf(1 - (alpha / 2)) * std_error, value_est + norm.ppf(1 - (alpha / 2)) * std_error]

    return value_est, confidence_bounds

# the idea behind control variates:
# - select a variable for which we know the mean, that is highly correlated with the option payoff, in this case ST.
# - run a simulation and compare observed ST to E(ST).
# - if ST has overshot its expectation, clearly we had a certain amount of "luck" that allowed this to happen, so we adjust our simulated option
#   payoff by some multiple (Beta) of the difference of ST from its expectation, and vice versa.
# - i.e. X_control = X - Beta * (Y - E(Y))
# - while this is strange because it can allow for things like negative values for our new X_control, this is fine as it is the average of these
#   that matters, and this does not introduce bias: E(X_control) = E(X) - Beta * (E(Y) - E(Y)) = E(X).
# - we use Beta* = Cov(X, Y) / Var(Y), as this is the variance-minimising choice of Beta.

def MC_value_option_control(call:bool, n_paths:int, S0:float, K:float, r:float, vol:float, T:float, alpha:float=0.05, fixed_Z=None):
    terminal_prices = simulate_gbm_cont(
        n_paths=n_paths,
        drift=r,
        vol=vol,
        S0=S0,
        T=T,
        fixed_Z=fixed_Z
    )

    E_ST = S0 * np.exp(r * T)

    if call:
        payoffs = [max(p - K, 0) for p in terminal_prices]
    else:
        payoffs = [max(K - p, 0) for p in terminal_prices]

    beta = np.cov(payoffs, terminal_prices)[0, 1] / np.var(terminal_prices)

    payoffs_control = [payoffs[i] - beta * (terminal_prices[i] - E_ST) for i in range(n_paths)]

    discounted_payoffs_control = [np.exp(-r * T) * p for p in payoffs_control]

    value_est = np.mean(discounted_payoffs_control)

    std_error = np.std(discounted_payoffs_control) / np.sqrt(n_paths)

    confidence_bounds = [value_est - norm.ppf(1 - (alpha / 2)) * std_error, value_est + norm.ppf(1 - (alpha / 2)) * std_error]

    return value_est, confidence_bounds

def MC_value_option_antithetic_control(call:bool, n_paths:int, S0:float, K:float, r:float, vol:float, T:float, alpha:float=0.05):
    terminal_prices = simulate_gbm_antithetic(
        n_paths=n_paths,
        drift=r,
        vol=vol,
        S0=S0,
        T=T,
    )

    terminal_prices = np.array(terminal_prices)

    if call:
        payoffs = np.maximum(terminal_prices - K, 0)
    else:
        payoffs = np.maximum(K - terminal_prices, 0)

    pair_payoffs = (payoffs[0::2] + payoffs[1::2]) / 2

    pair_prices = (terminal_prices[0::2] + terminal_prices[1::2]) / 2

    E_ST = S0 * np.exp(r * T)

    beta = (np.cov(pair_payoffs, pair_prices, ddof=1)[0, 1] / np.var(pair_prices, ddof=1))

    adjusted_pair_payoffs = (pair_payoffs - beta * (pair_prices - E_ST))

    discounted = np.exp(-r * T) * adjusted_pair_payoffs

    value_est = np.mean(discounted)

    std_error = (np.std(discounted, ddof=1) / np.sqrt(n_paths / 2))

    confidence_bounds = [value_est - norm.ppf(1 - (alpha / 2)) * std_error, value_est + norm.ppf(1 - (alpha / 2)) * std_error]

    return value_est, confidence_bounds

def MC_value_asian_option(call:bool, n_paths:int, S0:float, K:float, r:float, vol:float, T:float, alpha:float=0.05):
    price_paths = simulate_gbm(
        n_paths=n_paths,
        drift=r,
        vol=vol,
        S0=S0,
        T=T,
    )

    if call:
        payoffs = [max(np.mean(S[1:]) - K, 0) for S in price_paths]
    else:
        payoffs = [max(K - np.mean(S[1:]), 0) for S in price_paths]
    
    discounted_payoffs = [np.exp(-r * T) * p for p in payoffs]

    value_est = np.mean(discounted_payoffs)

    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)

    confidence_bounds = [value_est - norm.ppf(1 - (alpha / 2)) * std_error, value_est + norm.ppf(1 - (alpha / 2)) * std_error]

    return value_est, confidence_bounds
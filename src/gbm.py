import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Recall GBM: dS = muSdt + sigmaSdW, where dW = sqrt(dt)Z, Z ~ N(0, 1)
def simulate_gbm(n_paths:int, S0:float, drift:float, vol:float, T:float, dt:float=1/252):
    n_steps = int(T / dt)
    price_paths = []

    for _ in range(n_paths):
        S_path = [S0]

        for _ in range(n_steps):
            deterministic_increment = drift * S_path[-1] * dt

            z = np.random.normal(0, 1)
            dW = np.sqrt(dt) * z
            stochastic_increment = vol * S_path[-1] * dW

            price_increment = deterministic_increment + stochastic_increment

            S_path.append(S_path[-1] + price_increment)
    
        price_paths.append(S_path)

    return price_paths

def simulate_gbm_cont(n_paths:int, S0:float, drift:float, vol:float, T:float, fixed_Z=None):
    terminal_prices = []

    for i in range(n_paths):
        if fixed_Z is None:
            Z = np.random.normal(0, 1)
        else:
            Z = fixed_Z[i]

        terminal_price = S0 * np.exp((drift - (vol**2) / 2) * T + vol * np.sqrt(T) * Z)

        terminal_prices.append(terminal_price)

    return terminal_prices

# varaince reduction - antithetic variates
def simulate_gbm_antithetic(n_paths:int, S0:float, drift:float, vol:float, T:float):
    terminal_prices = []

    for _ in range(n_paths // 2):
        Z = np.random.normal(0, 1)
        terminal_price_plus = S0 * np.exp((drift - (vol**2) / 2) * T + vol * np.sqrt(T) * Z)
        terminal_price_minus = S0 * np.exp((drift - (vol**2) / 2) * T + vol * np.sqrt(T) * -Z)

        terminal_prices.append(terminal_price_plus)
        terminal_prices.append(terminal_price_minus)

    return terminal_prices
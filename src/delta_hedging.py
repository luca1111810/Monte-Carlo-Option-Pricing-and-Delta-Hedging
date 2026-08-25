import numpy as np
from scipy.stats import norm

from black_scholes import BlackScholes
from gbm import simulate_gbm
from monte_carlo import MC_value_option

# finite differencing: delta = (V(S0 + h) - V(S0 - h)) / 2h
def MC_delta(call:bool, n_paths:int, S0:float, K:float, r:float, vol:float, T:float, h:float=0.01):
    fixed_Z = np.random.normal(0, 1, n_paths)

    V_plus, _ = MC_value_option(
        call=call,
        n_paths=n_paths,
        S0=S0 + h,
        K=K,
        r=r,
        vol=vol,
        T=T,
        fixed_Z=fixed_Z
    )
    
    V_minus, _ = MC_value_option(
        call=call,
        n_paths=n_paths,
        S0=S0 - h,
        K=K,
        r=r,
        vol=vol,
        T=T,
        fixed_Z=fixed_Z
    )

    delta_est = (V_plus - V_minus) / (2 * h)

    return delta_est

# discrete delta hedging
def discrete_delta_hedge(call:bool, n_sims:int, S0:float, K:float, r:float, vol:float, T:float, dt:float=1/252, hedging_interval=5, trading_cost_prop:float=0):
    price_paths = simulate_gbm(
        n_paths=n_sims,
        drift=r,
        vol=vol,
        S0=S0,
        T=T,
        dt=dt
    )

    hedge_pnls = []
    trading_costs = []

    for S_path in price_paths:

        prev_option_value, prev_delta = BlackScholes(
            call=call,
            S0=S_path[0],
            K=K,
            r=r,
            vol=vol,
            T=T
        )

        V0 = prev_option_value

        cash = prev_delta * S_path[0] - prev_option_value - trading_cost_prop * S_path[0] * abs(prev_delta)
        trading_cost = trading_cost_prop * S_path[0] * abs(prev_delta)

        prev_i = 0

        for i in range(hedging_interval, len(S_path) - 1, hedging_interval):
            _, delta = BlackScholes(
                call=call,
                S0=S_path[i],
                K=K,
                r=r,
                vol=vol,
                T=T - (i * dt)
            )

            cash = cash * np.exp(r * hedging_interval * dt) + S_path[i] * (delta - prev_delta) - trading_cost_prop * S_path[i] * abs(delta - prev_delta)
            trading_cost += trading_cost_prop * S_path[i] * abs(delta - prev_delta)

            prev_delta = delta
            prev_i = i

        cash *= np.exp(r * (len(S_path) - 1 - prev_i) * dt)
        cash -= trading_cost_prop * S_path[-1] * abs(prev_delta)

        trading_cost += trading_cost_prop * S_path[-1] * abs(prev_delta)

        if call:
            pnl = max(S_path[-1] - K, 0) - (prev_delta * S_path[-1]) + cash
        else:
            pnl = max(K - S_path[-1], 0) - (prev_delta * S_path[-1]) + cash

        hedge_pnls.append(pnl)
        trading_costs.append(trading_cost)

    scaled_trading_costs = list(100 * np.array(trading_costs) / V0)

    return hedge_pnls, scaled_trading_costs
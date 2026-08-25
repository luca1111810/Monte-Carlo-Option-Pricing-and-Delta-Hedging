# Monte Carlo Option Pricing and Delta Hedging

A Python project exploring Monte Carlo methods for option pricing and hedging under the Black-Scholes framework.

The project develops reusable pricing and simulation functions alongside notebooks investigating the theoretical and numerical behaviour of the methods.

## Features

- Geometric Brownian motion simulation
- European option pricing using Monte Carlo simulation
- Validation against Black-Scholes analytical prices
- Monte Carlo convergence analysis
- Antithetic variates
- Control variates
- Arithmetic Asian option pricing
- Monte Carlo delta estimation using finite differences and common random numbers
- Discrete delta hedging
- Transaction costs and the hedge-frequency bias-variance tradeoff

## Structure

```text
notebooks/
    01_gbm_path_simulation.ipynb
    02_monte_carlo_option_valuation.ipynb
    03_monte_carlo_variance_reduction.ipynb
    04_asian_option_valuation.ipynb
    05_monte_carlo_delta.ipynb
    06_discrete_delta_hedging.ipynb

src/
    black_scholes.py
    delta_hedging.py
    gbm.py
    monte_carlo.py
```

The notebooks contain the numerical experiments, visualisations and theoretical discussion, while reusable functions are contained in `src/`.

## Method

Under risk-neutral geometric Brownian motion,

$$
dS_t = rS_t\,dt + \sigma S_t\,dW_t.
$$

Option values are estimated using

$$
V_0 = e^{-rT}E[H_T].
$$

Monte Carlo estimates are compared against Black-Scholes prices where an analytical benchmark is available.

Variance reduction techniques are used to improve estimator efficiency without changing the underlying pricing model.

The final section investigates discrete delta hedging and the tradeoff between replication error and transaction costs as hedge frequency changes.

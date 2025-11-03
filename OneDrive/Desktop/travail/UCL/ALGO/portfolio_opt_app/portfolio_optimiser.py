import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

st.title("Simple Portfolio Optimizer")

# --- User inputs ---
tickers_input = st.text_input("Enter stock tickers separated by commas", "AAPL,MSFT,GOOGL,AMZN")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

region = st.selectbox("Select Risk-Free Rate Region", ["US", "Belgium", "Europe", "Custom rate"])

try:
    if region == "US":
        rf_data = yf.download("^IRX", period="1d")
        if not rf_data.empty:
            rf = rf_data["Close"].iloc[-1] / 100  # IRX in %, convert to decimal
        else:
            rf = 0.02
    elif region == "Belgium":
        rf = 0.02
    elif region == "Europe":
        rf = 0.02
    elif region == "Custom rate":
        rf = st.number_input("Enter custom risk-free rate (as decimal)", 0.0, 0.1, 0.02)
    else:
        rf = 0.02
except:
    rf = 0.02

st.write("Using risk-free rate", rf)

opt_sharpe = st.checkbox("Optimize Sharpe Ratio")
opt_risk_pref = st.checkbox("Optimize Risk Preference")

if not tickers:
    st.write("Please enter at least one valid ticker")
    st.stop()

# --- Download stock data ---
price_data = yf.download(tickers, period="1y")["Adj Close"].dropna()

if price_data.empty:
    st.write("No data found for given tickers")
    st.stop()

returns = price_data.pct_change().dropna()
mean_returns = returns.mean()
covariance = returns.cov()

# --- Helper functions ---

def portfolio_perf(weights, returns, cov):
    port_return = np.dot(weights, returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return port_return, port_vol

def neg_sharpe(weights, returns, cov, rf):
    ret, vol = portfolio_perf(weights, returns, cov)
    return -(ret - rf) / vol if vol != 0 else 1e10

# --- Optimization setup ---

num_assets = len(mean_returns)
bounds = tuple((0, 1) for _ in range(num_assets))
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
start_weights = np.array(num_assets * [1. / num_assets])

if opt_sharpe:
    res_sharpe = minimize(
        neg_sharpe, start_weights, args=(mean_returns.values, covariance.values, rf),
        bounds=bounds, constraints=constraints, method='SLSQP'
    )
    weights_sharpe = res_sharpe.x
    ret_sharpe, vol_sharpe = portfolio_perf(weights_sharpe, mean_returns.values, covariance.values)
    sharpe_sharpe = (ret_sharpe - rf) / vol_sharpe if vol_sharpe != 0 else 0
else:
    weights_sharpe = None

if opt_risk_pref and not opt_sharpe:
    # Simple risk preference: minimize volatility (risk) for a target return
    target_return = st.slider("Target portfolio return", float(mean_returns.min()), float(mean_returns.max()), float(mean_returns.mean()))

    def portfolio_vol(weights):
        return portfolio_perf(weights, mean_returns.values, covariance.values)[1]

    constraints_risk = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns.values) - target_return}
    )

    res_risk = minimize(
        portfolio_vol, start_weights, bounds=bounds,
        constraints=constraints_risk, method='SLSQP'
    )
    weights_risk = res_risk.x
    ret_risk, vol_risk = portfolio_perf(weights_risk, mean_returns.values, covariance.values)
    sharpe_risk = (ret_risk - rf) / vol_risk if vol_risk != 0 else 0

if opt_sharpe and opt_risk_pref:
    # Move along CML by mixing tangency portfolio with risk-free asset
    risk_pref = st.slider("Select risky asset allocation", 0.0, 1.0, 1.0)

    # Use tangency portfolio weights
    if weights_sharpe is not None:
        w_risky_scaled = weights_sharpe * risk_pref
        w_rf = 1 - risk_pref

        port_return = ret_sharpe * risk_pref + rf * w_rf
        port_vol = vol_sharpe * risk_pref
        sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0

        st.write("Portfolio return", port_return)
        st.write("Portfolio volatility", port_vol)
        st.write("Sharpe ratio", sharpe)
        st.write("Weights in risky assets")
        st.write(pd.Series(w_risky_scaled, index=mean_returns.index))
        st.write("Weight in risk-free asset", w_rf)
    else:
        st.write("Error: Sharpe optimization not done")
elif opt_sharpe:
    st.write("Portfolio return", ret_sharpe)
    st.write("Portfolio volatility", vol_sharpe)
    st.write("Sharpe ratio", sharpe_sharpe)
    st.write("Weights")
    st.write(pd.Series(weights_sharpe, index=mean_returns.index))
elif opt_risk_pref:
    st.write("Portfolio return", ret_risk)
    st.write("Portfolio volatility", vol_risk)
    st.write("Sharpe ratio", sharpe_risk)
    st.write("Weights")
    st.write(pd.Series(weights_risk, index=mean_returns.index))
else:
    st.write("Please select at least one optimization method")
import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

st.title("Portfolio Optimiser")

# Dates for data
enddate = dt.datetime.now()
years = st.number_input("years analyzed:", min_value=1, value=5)
startdate = enddate - dt.timedelta(days=365 * years)

# Session state for portfolio tickers
if 'tickers' not in st.session_state:
    st.session_state['tickers'] = []

new_ticker = st.text_input("Enter a ticker").upper()

if st.button("Add stock") and new_ticker:
    if new_ticker not in st.session_state['tickers']:
        st.session_state['tickers'].append(new_ticker)
    else:
        st.warning(new_ticker + " is already in the portfolio.")

st.write("Portfolio:")
st.write(st.session_state['tickers'])

rem = st.selectbox("Remove a stock", [""] + st.session_state['tickers'])
if st.button("Remove") and rem:
    st.session_state['tickers'].remove(rem)

allow_shorting = st.checkbox("Allow shorting", value=False)

# --- Risk-free rate ---
region = st.selectbox("Select Risk-Free Rate Region", ["US", "Belgium", "Europe", "Custom rate"])
try:
    if region == "US":
        rf_data = yf.download("^IRX", period="1d")
        if not rf_data.empty:
            rf = rf_data["Close"].iloc[-1] / 100
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

st.write("Using risk-free rate:")
st.write(rf)

# Optimization style selector
opt_styles = st.multiselect("Select optimization methods:", ["Maximize Sharpe Ratio", "Optimize for Risk Preference"], default=["Maximize Sharpe Ratio"])

risk_aversion = 0.0
if "Optimize for Risk Preference" in opt_styles:
    risk_aversion = st.slider("Risk aversion", 0.0, 10.0, 3.0)

if len(st.session_state['tickers']) == 0:
    st.warning("Portfolio is empty")
    st.stop()

# Download price data
data = yf.download(st.session_state['tickers'], start=startdate, end=enddate)
close = data['Close'].copy()

# Calculate log returns and annualized stats
log_returns = np.log(close / close.shift(1)).dropna()
mean_returns = log_returns.mean() * 252
covariance = log_returns.cov() * 252

# Prepare tickers list
tickers = st.session_state['tickers'].copy()

# If both optimization methods selected, include risk-free asset and move along CML
include_rf = False
if "Maximize Sharpe Ratio" in opt_styles and "Optimize for Risk Preference" in opt_styles:
    include_rf = True

if include_rf:
    tickers.append("RISK_FREE")
    mean_returns = pd.concat([mean_returns, pd.Series([rf], index=["RISK_FREE"])])    # Extend covariance matrix for risk-free asset
    cov_rf_row = pd.Series(0, index=mean_returns.index)
    covariance = pd.concat([covariance, pd.DataFrame([cov_rf_row], index=["RISK_FREE"])])
    cov_rf_col = pd.Series(0, index=mean_returns.index)
    covariance["RISK_FREE"] = cov_rf_col
    covariance.loc["RISK_FREE", "RISK_FREE"] = 0.0

# Reindex mean_returns and covariance to match tickers
mean_returns = mean_returns.loc[tickers]
covariance = covariance.loc[tickers, tickers]

num_assets = len(tickers)

# Define portfolio performance function
def portfolio_perf(weights, returns, cov):
    port_return = np.dot(weights, returns)
    port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    return port_return, port_vol

# Sharpe ratio negative for minimization
def neg_sharpe(weights, returns, cov, rf):
    port_return, port_vol = portfolio_perf(weights, returns, cov)
    if port_vol == 0:
        return 1e10
    return -(port_return - rf) / port_vol

# Utility function for risk preference optimization (negative for minimization)
def neg_utility(weights, returns, cov, risk_aversion):
    port_return, port_vol = portfolio_perf(weights, returns, cov)
    utility = port_return - 0.5 * risk_aversion * (port_vol ** 2)
    return -utility

constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

if allow_shorting:
    bounds = tuple((-1, 1) for _ in range(num_assets))
else:
    bounds = tuple((0, 1) for _ in range(num_assets))

start = num_assets * [1.0 / num_assets]

# Convert mean_returns and covariance to numpy arrays
returns = mean_returns.values
cov = covariance.values

# Optimization logic
if include_rf:
    # Optimize tangency portfolio (max sharpe ratio)
    res = minimize(neg_sharpe, start, args=(returns, cov, rf), bounds=bounds, constraints=constraints, method='SLSQP')
    weights = res.x
    port_return, port_vol = portfolio_perf(weights, returns, cov)
    sharpe = (port_return - rf) / port_vol if port_vol != 0 else None

    # Move along CML based on risk preference (risk_aversion)
    if "Optimize for Risk Preference" in opt_styles:
        # Adjust weights to reflect desired risk preference by scaling weights with risk aversion
        # Calculate risk tolerance as 1/risk_aversion (avoid div by zero)
        risk_tolerance = 1.0 / risk_aversion if risk_aversion != 0 else 1e10
        # Weight on risky portfolio
        w_risky = risk_tolerance * (port_return - rf) / (port_vol ** 2)
        # Weight on risk-free asset
        w_rf = 1 - w_risky
        # Combine weights
        weights = np.append(weights[:-1] * w_risky, w_rf)
        # Recalculate portfolio stats
        port_return, port_vol = portfolio_perf(weights, returns, cov)
        sharpe = (port_return - rf) / port_vol if port_vol != 0 else None
else:
    if "Maximize Sharpe Ratio" in opt_styles:
        res = minimize(neg_sharpe, start, args=(returns, cov, rf), bounds=bounds, constraints=constraints, method='SLSQP')
        weights = res.x
        port_return, port_vol = portfolio_perf(weights, returns, cov)
        sharpe = (port_return - rf) / port_vol if port_vol != 0 else None
    elif "Optimize for Risk Preference" in opt_styles:
        res = minimize(neg_utility, start, args=(returns, cov, risk_aversion), bounds=bounds, constraints=constraints, method='SLSQP')
        weights = res.x
        port_return, port_vol = portfolio_perf(weights, returns, cov)
        sharpe = (port_return - rf) / port_vol if port_vol != 0 else None
    else:
        st.warning("Please select at least one optimization method.")
        st.stop()

# Display results
st.subheader("Optimized Portfolio Weights")
weights_table = pd.DataFrame({"Ticker": tickers, "Weight": weights}).set_index("Ticker")
st.write(weights_table)

st.write("Expected Annual Return:")
st.write(port_return)

st.write("Expected Annual Volatility:")
st.write(port_vol)

if sharpe is not None:
    st.write("Sharpe Ratio:")
    st.write(sharpe)

# Plot weights
fig, ax = plt.subplots()
weights_table['Weight'].plot(kind='bar', ax=ax)
ax.set_title("Optimal Portfolio Weights")
ax.set_ylabel("Weight")
st.pyplot(fig)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from scipy.optimize import minimize
import matplotlib.pyplot as plt

st.title("Portfolio Optimiser")

# --- Dates and portfolio input ---
enddate = dt.datetime.now()
years = st.number_input("Years analyzed:", min_value=1, value=5)
startdate = enddate - dt.timedelta(days=365 * years)

if 'tickers' not in st.session_state:
    st.session_state['tickers'] = []

new_ticker = st.text_input("Enter a ticker").upper()
if st.button("Add stock") and new_ticker:
    if new_ticker not in st.session_state['tickers']:
        st.session_state['tickers'].append(new_ticker)
    else:
        st.warning(f"{new_ticker} is already in the portfolio.")
st.write("Portfolio:", st.session_state['tickers'])

rem = st.selectbox("Remove a stock", [""] + st.session_state['tickers'])
if st.button("Remove") and rem:
    st.session_state['tickers'].remove(rem)

if len(st.session_state['tickers']) == 0:
    st.warning("Portfolio is empty")
    st.stop()

allow_shorting = st.checkbox("Allow shorting", value=False)

# --- Risk-free rate ---
region = st.selectbox("Select Risk-Free Rate Region", ["US", "Belgium", "Europe", "Custom rate"])
try:
    if region == "US":
        rf_data = yf.download("^IRX", period="1d")
        rf = rf_data["Close"].iloc[-1] / 100 if not rf_data.empty else 0.02
    elif region == "Belgium":
        rf = 0.02  # placeholder
    elif region == "Europe":
        rf = 0.02  # placeholder
    else:
        rf = st.number_input("Enter custom risk-free rate (as decimal)", 0.0, 0.1, 0.02)
except:
    rf = 0.02
st.write(f"Using risk-free rate: {rf:.4f}")

# --- Optimization selection ---
opt_styles = st.multiselect("Select optimization methods:", ["Maximize Sharpe Ratio", "Optimize for Risk Preference"], default=["Maximize Sharpe Ratio"])

risk_aversion = 3.0
if "Optimize for Risk Preference" in opt_styles:
    risk_aversion = st.slider("Risk aversion (higher = more conservative)", 0.0, 10.0, 3.0)

# --- Download data ---
data = yf.download(st.session_state['tickers'], start=startdate, end=enddate)['Close']
log_returns = np.log(data / data.shift(1)).dropna()
mean_returns = log_returns.mean() * 252
covariance = log_returns.cov() * 252

tickers = st.session_state['tickers']
num_assets = len(tickers)

# --- Helper functions ---
def portfolio_perf(weights, returns, cov):
    port_return = np.dot(weights, returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return port_return, port_vol

def neg_sharpe(weights, returns, cov, rf):
    r, vol = portfolio_perf(weights, returns, cov)
    return -(r - rf) / vol

def neg_utility(weights, returns, cov, risk_aversion):
    r, vol = portfolio_perf(weights, returns, cov)
    return -(r - (risk_aversion / 2) * vol**2)

constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(-1, 1) if allow_shorting else (0, 1) for _ in range(num_assets)]
start = num_assets * [1. / num_assets]

# --- Optimization logic ---
if "Maximize Sharpe Ratio" in opt_styles and "Optimize for Risk Preference" in opt_styles:
    # 1) Find tangency portfolio (max Sharpe)
    res = minimize(neg_sharpe, start, args=(mean_returns.values, covariance.values, rf), bounds=bounds, constraints=constraints, method='SLSQP')
    tangency_weights = res.x
    tangency_return, tangency_vol = portfolio_perf(tangency_weights, mean_returns.values, covariance.values)

    # 2) Combine with risk-free asset according to risk preference (risk_aversion)
    # Map risk_aversion to weight in tangency portfolio between 0 and 1 (simple linear inverse)
    w_p = max(0, min(1, 1 - risk_aversion / 10))  # risk_aversion=0 -> w_p=1 (all risky), 10->w_p=0 (all rf)
    w_rf = 1 - w_p

    port_return = w_rf * rf + w_p * tangency_return
    port_vol = w_p * tangency_vol
    sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0

    # Display weights including risk-free
    weights_df = pd.DataFrame({
        'Ticker': tickers + ['RISK_FREE'],
        'Weight': np.append(w_p * tangency_weights, w_rf)
    }).set_index('Ticker')

    st.subheader("Portfolio on the Capital Market Line (CML)")
    st.write(weights_df)
    st.write(f"Expected Annual Return: {port_return:.2%}")
    st.write(f"Expected Annual Volatility: {port_vol:.2%}")
    st.write(f"Sharpe Ratio: {sharpe:.2f}")

elif "Maximize Sharpe Ratio" in opt_styles:
    # Just tangency portfolio (risky assets only)
    res = minimize(neg_sharpe, start, args=(mean_returns.values, covariance.values, rf), bounds=bounds, constraints=constraints, method='SLSQP')
    weights = res.x
    ret, vol = portfolio_perf(weights, mean_returns.values, covariance.values)
    sharpe = (ret - rf) / vol if vol > 0 else 0

    weights_df = pd.DataFrame({'Ticker': tickers, 'Weight': weights}).set_index('Ticker')

    st.subheader("Tangency Portfolio (Max Sharpe Ratio)")
    st.write(weights_df)
    st.write(f"Expected Annual Return: {ret:.2%}")
    st.write(f"Expected Annual Volatility: {vol:.2%}")
    st.write(f"Sharpe Ratio: {sharpe:.2f}")

elif "Optimize for Risk Preference" in opt_styles:
    # Efficient frontier portfolios (no risk-free)
    res = minimize(neg_utility, start, args=(mean_returns.values, covariance.values, risk_aversion), bounds=bounds, constraints=constraints, method='SLSQP')
    weights = res.x
    ret, vol = portfolio_perf(weights, mean_returns.values, covariance.values)
    sharpe = (ret - rf) / vol if vol > 0 else 0

    weights_df = pd.DataFrame({'Ticker': tickers, 'Weight': weights}).set_index('Ticker')

    st.subheader("Portfolio Optimized for Risk Preference")
    st.write(weights_df)
    st.write(f"Expected Annual Return: {ret:.2%}")
    st.write(f"Expected Annual Volatility: {vol:.2%}")
    st.write(f"Sharpe Ratio: {sharpe:.2f}")

# --- Plot weights ---
fig, ax = plt.subplots()
weights_df['Weight'].plot(kind='bar', ax=ax)
ax.set_ylabel('Weight')
ax.set_title('Portfolio Weights')
st.pyplot(fig)
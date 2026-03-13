import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


st.title("Portfolio Optimiser")


# ---------- Time period ----------
enddate = dt.datetime.now()
years = st.number_input("Years analysed", min_value=1, value=5)
startdate = enddate - dt.timedelta(days=365 * years)


# ---------- Portfolio builder ----------
if "tickers" not in st.session_state:
    st.session_state["tickers"] = []

new_ticker = st.text_input("Add ticker").upper()

if st.button("Add stock") and new_ticker:
    if new_ticker not in st.session_state["tickers"]:
        st.session_state["tickers"].append(new_ticker)

rem = st.selectbox("Remove ticker", [""] + st.session_state["tickers"])
if st.button("Remove") and rem:
    st.session_state["tickers"].remove(rem)

st.write("Portfolio:", st.session_state["tickers"])



# ---------- Risk-free rate ----------
def get_rf_rate(ticker):
    try:
        rf_data = yf.download(ticker, period="5d", progress=False)
        if not rf_data.empty:
            return float(rf_data["Close"].iloc[-1]) / 100
    except:
        pass
    return 0.02


region = st.selectbox(
    "Risk-free rate region",
    ["US", "Europe", "Custom"]
)

if region == "US":
    rf = get_rf_rate("^IRX")
elif region == "Europe":
    rf = get_rf_rate("^TNX")
elif region == "Custom":
    rf = st.number_input("Custom risk-free rate", value=0.02)
else:
    rf = 0.02

st.write(f"Using risk-free rate: {rf:.4f}")

# ---------- Optimisation style ----------
opt_style = st.radio("Optimisation objective",("Maximize Sharpe Ratio","Risk Preference","Both (Capital Allocation Line)")
)

allow_shorting = st.checkbox("Allow shorting")

risk_aversion = None
if opt_style in ["Risk Preference", "Both (Capital Allocation Line)"]:
    risk_aversion = st.slider("Risk aversion", 0.1, 10.0, 3.0)


if len(st.session_state["tickers"]) == 0:
    st.warning("Portfolio is empty")
    st.stop()


# ---------- Download price data ----------
data = yf.download(
    st.session_state["tickers"],
    start=startdate,
    end=enddate,
    progress=False
)

close = data["Close"]


# ---------- Returns ----------
log_returns = np.log(close / close.shift(1)).dropna()
mean_returns = log_returns.mean() * 252


# ---------- Covariance (Ledoit-Wolf) ----------
lw = LedoitWolf()
lw.fit(log_returns)

covariance = lw.covariance_ * 252


tickers = st.session_state["tickers"]
num_stocks = len(tickers)


# ---------- Portfolio math ----------
def portfolio_perf(weights, returns, cov):
    ret = np.dot(weights, returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return ret, vol


def sharpe_objective(w, returns, cov, rf):
    r, v = portfolio_perf(w, returns, cov)
    return -(r - rf) / v


def utility_objective(w, returns, cov, A):
    r, v = portfolio_perf(w, returns, cov)
    return -(r - (A/2)*(v**2))


constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

if allow_shorting:
    bounds = tuple((-1, 1) for _ in range(num_stocks))
else:
    bounds = tuple((0, 1) for _ in range(num_stocks))

start = num_stocks * [1/num_stocks]


# ---------- Tangency portfolio ----------
tangency = minimize(
    sharpe_objective,
    start,
    args=(mean_returns.values, covariance, rf),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

w_tan = tangency.x
ret_tan, vol_tan = portfolio_perf(w_tan, mean_returns.values, covariance)


# ---------- Optimisation ----------
if opt_style == "Maximize Sharpe Ratio":

    optimal_weights = w_tan
    opt_return, opt_vol = ret_tan, vol_tan


elif opt_style == "Risk Preference":

    res = minimize(
        utility_objective,
        start,
        args=(mean_returns.values, covariance, risk_aversion),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = res.x
    opt_return, opt_vol = portfolio_perf(
        optimal_weights, mean_returns.values, covariance
    )


elif opt_style == "Both (Capital Allocation Line)":

    w_t = (ret_tan - rf) / (risk_aversion * vol_tan**2)
    w_rf = 1 - w_t

    optimal_weights = w_t * w_tan
    opt_return = w_rf * rf + w_t * ret_tan
    opt_vol = abs(w_t) * vol_tan


# ---------- Results ----------
st.subheader("Optimised Weights")

weights_df = pd.DataFrame({
    "Ticker": tickers,
    "Weight": optimal_weights
}).set_index("Ticker")

st.write(weights_df)

st.write(f"Expected Return: {opt_return:.2%}")
st.write(f"Volatility: {opt_vol:.2%}")

if opt_style == "Maximize Sharpe Ratio":
    sharpe = (opt_return - rf) / opt_vol
    st.write(f"Sharpe Ratio: {sharpe:.2f}")


# ---------- Bar chart --------
fig, ax = plt.subplots()
weights_df["Weight"].plot(kind="bar", ax=ax)
ax.set_title("Portfolio Weights")
ax.set_ylabel("Weight")
st.pyplot(fig)

# ------- Efficient frontier ----------
def efficient_frontier(returns, cov, points=100, allow_shorting=False):
    num_assets = len(returns)
    results = []
    
    # Target returns for frontier
    target_returns = np.linspace(min(returns), max(returns)*1.5, points)
    
    bounds = [(-1,1)]*num_assets if allow_shorting else [(0,1)]*num_assets
    constraints_base = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    
    for r_target in target_returns:
        # Constraint: portfolio return = r_target
        constraints = constraints_base + [{"type": "eq", "fun": lambda w, r_target=r_target: np.dot(w, returns) - r_target}]
        res = minimize(lambda w: np.dot(w, np.dot(cov, w)),  # Minimize variance
                       num_assets*[1/num_assets],
                       method="SLSQP",
                       bounds=bounds,
                       constraints=constraints)
        if res.success:
            vol = np.sqrt(np.dot(res.x, np.dot(cov, res.x)))
            results.append((vol, r_target))
    
    vols, rets = zip(*results)
    return np.array(vols), np.array(rets)

frontier_vol, frontier_ret = efficient_frontier(mean_returns.values, covariance, allow_shorting=allow_shorting)

# -------- Plot ---------
fig, ax = plt.subplots(figsize=(10,6))

# Efficient frontier
ax.plot(frontier_vol, frontier_ret, color='#0b3d91', linewidth=3, label='Efficient Frontier')

# CAL
cal_x = np.linspace(0, max(frontier_vol)*1.2, 100)
cal_y = rf + (ret_tan - rf)/vol_tan * cal_x
ax.plot(cal_x, cal_y, color='r', linestyle='--', linewidth=2, label='Capital Allocation Line')

# Tangency portfolio
ax.scatter(vol_tan, ret_tan, marker='*', color='r', s=150, label='Tangency Portfolio')

#Optimised Portfolio
ax.scatter(opt_vol, opt_return, marker='D', color='b', s=100, label='Our Portfolio')



legend = ax.legend(frameon=True, fontsize=10)
for handle in legend.legendHandles:
    handle.set_sizes([50])  # shrink scatter markers in legend

ax.set_title('Efficient Frontier with CAL')
ax.set_xlabel('Volatility (Std Dev)')
ax.set_ylabel('Expected Return')
ax.grid(True)
st.pyplot(fig)

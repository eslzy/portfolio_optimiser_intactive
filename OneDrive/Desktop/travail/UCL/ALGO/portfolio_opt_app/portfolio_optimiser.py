import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.title("Portfolio Optimiser")


enddate = dt.datetime.now()
years = st.number_input("years analyzed:", min_value=1, value=5)
startdate = enddate - dt.timedelta(days=365 * years)


# this is how well handle the portfolio creation, so we create a session state so that it doesnt clear everytime we change smth
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

allow_shorting = st.checkbox("Allow shorting", value=False)


# this is where they can select a risk free rate depending on region, i might add more, default set as 2%
region = st.selectbox("Select Risk-Free Rate Region", ["US", "Europe", "Custom rate"])

# robust RF fetch: if ticker fails, we fallback to default 2%
try:
    if region == "US":
        rf_data = yf.download("^IRX", period="1d")
#    elif region == "Belgium":
#        rf_data = yf.download("BELG_TICKER", period="1d")
    elif region == "Europe":
        rf_data = yf.download("^ITX", period="1d")
    else:
        rf_data = pd.DataFrame()

    if region == "Custom rate":
        # number_input doesn't support placeholder; use a default value instead
        rf = st.number_input("Enter custom risk-free rate (decimal, 0.02 for 2%)", value=0.02, step=0.0005) #step??
    elif not rf_data.empty and "Close" in rf_data.columns:
        rf = float(rf_data["Close"].iloc[-1]) / 100.0
    else:
        st.warning("Error, defaulting to 2%.")
        rf = 0.02
except Exception as e:
    st.warning(f"Error fetching risk-free rate: {e}. Defaulting to 2%.")
    rf = 0.02

# idek if its worth having a region selection, idk how relevant it'll be in this context and idk how to handle multi currency portfolios
st.write(f"Using risk-free rate: {float(rf):.4f}")


# different optimisation options (now multi-select so user can pick one or both)
opt_styles = st.multiselect(
    "Select optimization methods:",
    ["Maximize Sharpe Ratio", "Optimize for Risk Preference"],
    default=["Maximize Sharpe Ratio"]
)

# risk aversion slider (we only show if user selects risk preference — but we set var now)
risk_aversion = 3.0
if "Optimize for Risk Preference" in opt_styles:
    risk_aversion = st.slider("Risk aversion (higher = more risk averse)", 0.0, 10.0, 3.0)


# allow including RF asset in the *universe* (independent of chosen optimization methods)
include_rf = st.checkbox("Include risk-free asset as an option in portfolio", value=False)


if len(st.session_state['tickers']) == 0:
    st.warning("Portfolio is empty")
    st.stop()

# download price data once (single call)
data = yf.download(st.session_state['tickers'], start=startdate, end=enddate)
# If yfinance returns a multi-index (group_by ticker) accessing ['Close'] works; otherwise it's a DataFrame with columns
# Keep existing approach assuming data['Close'] exists (we've used this earlier). If it fails, the app will raise there.
close = data['Close'].copy()

# log returns and cov stuff, alr anualised
log_returns = np.log(close / close.shift(1)).dropna()
covariance = log_returns.cov() * 252
mean_an_returns = log_returns.mean() * 252

# create working copies (we may augment them with RISK_FREE if requested)
mean_an = mean_an_returns.copy()
cov = covariance.copy()

# if user wants risk-free asset in universe, append it (zero vol, zero covariances)
if include_rf:
    # only append if not already present
    if "RISK_FREE" not in mean_an.index:
        mean_an = pd.concat([mean_an, pd.Series([rf], index=["RISK_FREE"])])
        # create zero row (index = current mean_an index AFTER we appended, but we want zeros for cov row with original assets)
        # build a zero row/col matching the final mean_an index
        zero_row = pd.Series(0.0, index=mean_an.index)
        # append row (as DataFrame) and then add column
        cov = pd.concat([cov, pd.DataFrame([zero_row], index=["RISK_FREE"])])
        cov["RISK_FREE"] = zero_row
        cov.loc["RISK_FREE", "RISK_FREE"] = 0.0

# tickers universe for optimization (copy of session tickers; we'll add RISK_FREE implicitly by checking mean_an index)
tickers = st.session_state['tickers'].copy()
if include_rf and "RISK_FREE" in mean_an.index:
    # the asset universe has the risk-free asset as well; we show it in outputs when used by specific optimization
    # note: we don't mutate session_state['tickers'] — just the working universe
    pass

# helper functions
def portfolio_perf(w, erm, cov):
    # w: 1D array, erm: 1D array of expected returns, cov: 2D covariance matrix
    returns = np.dot(w, erm)
    vol = np.sqrt(np.dot(w, np.dot(cov, w)))
    return returns, vol

def sharpe_opt(w, erm, cov, rf):
    port_return, port_vol = portfolio_perf(w, erm, cov)
    # guard against zero vol
    if port_vol == 0 or not np.isfinite(port_vol):
        return 1e9
    sharpe_ratio = (port_return - rf) / port_vol
    return -sharpe_ratio  # minimize negative Sharpe

def utility_opt(w, erm, cov, risk_aversion):
    port_return, port_vol = portfolio_perf(w, erm, cov)
    utility = port_return - (risk_aversion / 2.0) * (port_vol ** 2)  # return - (A/2)*variance
    return -utility  # minimize negative utility -> maximize utility


results = {}  # store results for each method

# ---------- Run Sharpe optimization if requested ----------
if "Maximize Sharpe Ratio" in opt_styles:
    # build the asset list for this optimization: include RISK_FREE if present in mean_an
    tickers_sharpe = tickers.copy()
    if "RISK_FREE" in mean_an.index and include_rf:
        tickers_sharpe = tickers_sharpe + ["RISK_FREE"]

    # build aligned erm and cov for the chosen tickers_sharpe
    erm_sharpe = mean_an.loc[tickers_sharpe].values
    cov_sharpe = cov.loc[tickers_sharpe, tickers_sharpe].values

    num_sharpe = len(tickers_sharpe)
    constraints_sharpe = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds_sharpe = tuple((-1, 1) if allow_shorting else (0, 1) for _ in range(num_sharpe))
    starting_sharpe = num_sharpe * [1.0 / num_sharpe]

    opt_sh = minimize(sharpe_opt, starting_sharpe, args=(erm_sharpe, cov_sharpe, rf),
                      method='SLSQP', bounds=bounds_sharpe, constraints=constraints_sharpe)

    if not opt_sh.success:
        st.warning(f"Sharpe optimization did not converge: {opt_sh.message}")
    weights_sharpe = opt_sh.x if opt_sh.success else starting_sharpe
    ret_sh, vol_sh = portfolio_perf(weights_sharpe, erm_sharpe, cov_sharpe)
    sharpe_val = (ret_sh - rf) / vol_sh if (vol_sh != 0 and np.isfinite(vol_sh)) else np.nan
    results["MaxSharpe"] = (tickers_sharpe, weights_sharpe, ret_sh, vol_sh, sharpe_val)


# ---------- Run Risk-preference optimization if requested ----------
if "Optimize for Risk Preference" in opt_styles:
    tickers_risk = tickers.copy()
    # If user included RISK_FREE in universe and wants it available, include it for risk optimization too
    if "RISK_FREE" in mean_an.index and include_rf:
        tickers_risk = tickers_risk + ["RISK_FREE"]

    erm_risk = mean_an.loc[tickers_risk].values
    cov_risk = cov.loc[tickers_risk, tickers_risk].values

    num_risk = len(tickers_risk)
    constraints_risk = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds_risk = tuple((-1, 1) if allow_shorting else (0, 1) for _ in range(num_risk))
    starting_risk = num_risk * [1.0 / num_risk]

    opt_r = minimize(utility_opt, starting_risk, args=(erm_risk, cov_risk, risk_aversion),
                     method='SLSQP', bounds=bounds_risk, constraints=constraints_risk)

    if not opt_r.success:
        st.warning(f"Risk-preference optimization did not converge: {opt_r.message}")
    weights_risk = opt_r.x if opt_r.success else starting_risk
    ret_r, vol_r = portfolio_perf(weights_risk, erm_risk, cov_risk)
    sharpe_r = (ret_r - rf) / vol_r if (vol_r != 0 and np.isfinite(vol_r)) else np.nan
    results["RiskPref"] = (tickers_risk, weights_risk, ret_r, vol_r, sharpe_r)


# ---------- Display results ----------
if len(results) == 0:
    st.warning("No optimization selected. Please choose at least one optimization method.")
    st.stop()

for key, (tkrs, wts, ret, vol, sh) in results.items():
    st.subheader(f"{key} Optimization Results")
    # create table
    weights_df = pd.DataFrame({"Ticker": tkrs, "Weight": wts}).set_index("Ticker")
    # format weights as percentages in a table for nicer display
    st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}))

    st.write(f"Expected Annual Return: {ret:.2%}")
    st.write(f"Expected Annual Volatility: {vol:.2%}")
    # always show Sharpe if computable
    if np.isfinite(sh):
        st.write(f"Sharpe Ratio: {sh:.2f}")
    else:
        st.write("Sharpe Ratio: N/A (couldn't compute)")

    # chart
    fig, ax = plt.subplots()
    weights_df['Weight'].plot(kind='bar', ax=ax)
    ax.set_title(f"{key} - Portfolio Weights")
    ax.set_ylabel("Weight")
    st.pyplot(fig)
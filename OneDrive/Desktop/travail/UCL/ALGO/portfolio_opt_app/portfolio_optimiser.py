import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from scipy.optimize import minimize

st.title("Simple Portfolio Optimizer")

enddate = dt.datetime.now()
years = st.number_input("years analyzed:", min_value=1, value=5)
startdate = enddate - dt.timedelta(days=365 * years)


#stock inputs
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


#rfr selection
region = st.selectbox("Select Risk-Free Rate Region", ["US", "Europe", "Custom rate"])

if region == "US":
    rf_data = yf.download("^IRX", period="1mo")["Close"]
    rf = float(rf_data.dropna().iloc[-1]) / 100  # last close, convert % to decimal
elif region == "Europe":
    rf_data = yf.download("^IRC", period="1mo")["Close"]  # Euro area long-term rate
    rf = float(rf_data.dropna().iloc[-1]) / 100
elif region == "Custom rate":
    rf = st.number_input("Enter custom annual risk-free rate (in decimal)", 0.02)
else:
    rf = 0.02
    


#option selection

opt_styles = st.multiselect("Select optimization methods:", ["Maximize Sharpe Ratio", "Optimize for Risk Preference"], default=["Maximize Sharpe Ratio"])
if not st.session_state['tickers'].len == 0:
    data = yf.download(st.session_state['tickers'], start=startdate, end=enddate)
else:
    st.write("Enter stocks in your portfolio")
close = data['Close'].copy()

# Calculate log returns and annualized stats
log_returns = np.log(close / close.shift(1)).dropna()
mean_returns = log_returns.mean() * 252
covariance = log_returns.cov() * 252

tickers = st.session_state['tickers'].copy()

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

allow_shorting = st.checkbox("Allow shorting", value=False)

#constraints
if allow_shorting:
    bounds = tuple((-1, 1) for _ in range(num_assets))
else:
    bounds = tuple((0, 1) for _ in range(num_assets))
start = num_assets * [1.0 / num_assets]


#------max sharpe opt--------------------------------------------------------------------------------------------------------------------------------------------------------

if "Maximize Sharpe Ratio" in opt_styles and "Optimize for Risk Preference" not in opt_styles:
    
    #Optimize (Sharpe ratio)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1} #in section bc we have diff constraints for other selections
    res = minimize(neg_sharpe, start, args=(mean_returns, covariance, rf), method='SLSQP', bounds=bounds,constraints=constraints)

    opt_weights = res.x
    (port_return, port_vol) = portfolio_perf(opt_weights, mean_returns, covariance)
    sharpe = (port_return - rf) / port_vol

    #results
    st.subheader("Optimal Tangency Portfolio (Max Sharpe Ratio)")
    st.write("Expected Annual Return:", port_return)
    st.write("Annual Volatility (Risk):", port_vol)
    st.write("Sharpe Ratio:", sharpe)
    st.write("Risk Free Rate:",rf)

    # Create and display a bar chart of weights
    weights_df = pd.DataFrame({'Ticker': tickers,'Weight': opt_weights})
    st.bar_chart(weights_df.set_index('Ticker'))

    # Optional: show numeric table too
    st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}))#pres techo 5nov 16:15 sale espa tu connais

#-------risk pref opt-------------------------------------------------------------------------------------------------------------------------------------------------------
elif "Optimize for Risk Preference" in opt_styles and "Maximize Sharpe Ratio" not in opt_styles:
    
    def portfolio_vol(weights, cov):
        return np.sqrt(np.dot(weights, np.dot(cov, weights)))
                                                                  #redid the f() bc the otehr one handles both erm and vol but we want to max only the return f() so redid one here 
    def neg_return(weights, returns):
        return -np.dot(weights, returns)
    
    target_risk = st.slider("Select target volatility (risk %):", min_value=0.01, max_value=0.70, value=0.15, step=0.01)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, {'type': 'eq', 'fun': lambda w: portfolio_vol(w, covariance) - target_risk}] #to make sure the portfolio has the desirred volatility
    
    res = minimize(neg_return, start, args=(mean_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    
    opt_weights = res.x
    port_return, port_vol = portfolio_perf(opt_weights, mean_returns, covariance)
    sharpe = (port_return - rf) / port_vol
    
    #results
    st.subheader("Risk Optimised Portfolio (Efficient Frontier)")
    st.write("Expected Annual Return:", port_return)
    st.write("Desired annual volatility:", target_risk)
    st.write("Annual Volatility (Risk):", port_vol)
    st.write("Sharpe Ratio:", sharpe)
    st.write("Risk Free Rate:",rf)

    # Create and display a bar chart of weights
    weights_df = pd.DataFrame({'Ticker': tickers,'Weight': opt_weights})
    st.bar_chart(weights_df.set_index('Ticker'))

    # Optional: show numeric table too
    st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}))
    

#--------------------------------------------------------------------------------------------------------------------------------------------------------------


else:
    st.write("Select optimisation type")



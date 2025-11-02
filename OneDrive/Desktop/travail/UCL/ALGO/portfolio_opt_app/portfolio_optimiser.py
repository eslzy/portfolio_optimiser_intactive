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


#this is how well handle the portfolio creation, so we create a session state so that it doesnt clear everytime we change smth
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

allow_shorting = st.checkbox("Allow shorting", value = False)


#this is where they can select a risk free rate depending on region, i might add more, default set as 2%
region = st.selectbox("Select Risk-Free Rate Region", ["US", "Belgium", "Europe", "Custom rate"])

#if it doesnt work well theres a plugin on github i foudn that facilitates yfinance rf tickers
#i saw a bunch of diff tickers to use so idk if theyre the right ones, will confirm later but its unclear so far 
try:
    if region == "US":
        rf_data = yf.download("^IRX", period="1d")
    elif region == "Belgium":
        rf_data = yf.download("BELG_TICKER", period="1d")
    elif region == "Europe":
        rf_data = yf.download("^ITX", period="1d")
    else:
        rf_data = pd.DataFrame()

    if not rf_data.empty and "Close" in rf_data.columns:
        rf = rf_data["Close"].iloc[-1] / 100
    else:
        st.warning("Could not fetch risk-free rate data. Defaulting to 2%.")
        rf = 0.02
except Exception as e:
    st.warning(f"Error fetching risk-free rate: {e}. Defaulting to 2%.")
    rf = 0.02
# idek if its worth having a region selection, idk how relevant it'll be in this context and idk how to handle multi currency portfolios

st.write(f"Using risk-free rate: {float(rf):.4f}")

#different optimisation types, idk if this is a good idea , complicates it a lot 
opt_style = st.radio("Select optimization opt_style:", ("Maximize Sharpe Ratio", "Optimize for Risk Preference"))

risk_aversion = 0.0
if opt_style == "Optimize for Risk Preference":
    risk_aversion = st.slider("Risk aversion", 0.0, 10.0, 3.0)


if len(st.session_state['tickers']) == 0:
    st.warning("Portfolio is empty")
    st.stop()

data = yf.download(st.session_state['tickers'], start=startdate, end=enddate)
close = data['Close'].copy()

#log returns and cov stuff, alr anualised 
log_returns = np.log(close / close.shift(1)).dropna()
covariance = log_returns.cov() * 252 
mean_an_returns = log_returns.mean() * 252 

#this part is annoying, it includes or not the rf rate depending on the selected optimisation type
#had to use some help to clean this up bc it just looked super messy
tickers = st.session_state['tickers'].copy()

if opt_style == "Maximize Sharpe Ratio":
    include_rf = st.checkbox("Include risk-free asset as an option in portfolio", value=False)
    if include_rf:
        tickers.append("RISK_FREE_ASSET")
        mean_an_returns = mean_an_returns.append(pd.Series(rf, index=["RISK_FREE"]))
        cov_rf_row = pd.Series(0, index=mean_an_returns.index)
        covariance = covariance.append(pd.DataFrame([cov_rf_row], index=["RISK_FREE"]))
        cov_rf_col = pd.Series(0, index=mean_an_returns.index)
        covariance["RISK_FREE"] = cov_rf_col
        covariance.loc["RISK_FREE", "RISK_FREE"] = 0.0

num_stocks = len(tickers)



def portfolio_perf(w, erm, cov):
    returns = np.dot(w, erm)
    vol = np.sqrt(np.dot(w, np.dot(cov, w))) #might have to transpose the weights depending on how we get the weights 
    return returns, vol

def sharpe_opt(w, erm, cov, rf):
    port_return, port_vol = portfolio_perf(w, erm, cov)
    sharpe_ratio = (port_return - rf) / port_vol
    return -sharpe_ratio   #we make it negative because we'll us ethe scipy.minimize f(), so by minimising the neg.sharpe, we are in terms maximising the sharpe

#had to add this for the case of the optimisation for risk pref opt. style. 
def utility_opt(w, erm, cov, risk_aversion):
    port_return, port_vol = portfolio_perf(w, erm, cov)
    utility = port_return - (risk_aversion / 2) * (port_vol ** 2) #return - (risk_aversion / 2) * variance
    return -utility  # same thing as the sharpe f(), we take the negative bc well minimise that negative == maximise


constraints = {'type':'eq', 'fun':lambda w: np.sum(w) - 1}  #this makes sure all the weights add to 1 (-1=0) lambda is like a quick write f() so lambda x: ___x == def__(x):___

if allow_shorting:
    bounds = tuple((-1, 1) for _ in range(num_stocks))  #if shorting's allowed we have to allow egative weights 
else:
    bounds = tuple((0, 1) for _ in range(num_stocks))

starting_point = num_stocks * [1 / num_stocks]


#this is the min formula its complicated
if opt_style == "Maximize Sharpe Ratio":
    opt_result = minimize(sharpe_opt, starting_point, args=(mean_an_returns.values, covariance.values, rf),method='SLSQP', bounds=bounds, constraints=constraints)
else:
    opt_result = minimize(utility_opt, starting_point, args=(mean_an_returns.values, covariance.values, risk_aversion),method='SLSQP', bounds=bounds, constraints=constraints)
# so the minimize f() takes in a function which it will minimize
#then it takes a startting point, it will go from there and then itteratively optimize, so this is the variables it has to optimize
#then it takes the rest of teh variables for the function which are fixed
#the SLSQP method is the sequential least squares method which is good for opt under constraint




optimal_weights = opt_result.x  #the minn f() returns an object, the x attribute is the opt variable
opt_return, opt_volatility = portfolio_perf(optimal_weights, mean_an_returns.values, covariance.values)



st.subheader("Optimized Portfolio Weights")

weights_table = pd.DataFrame({"Ticker": tickers,"Weight": optimal_weights}).set_index("Ticker")

st.write(weights_table)

st.write(f"Expected Annual Return: {opt_return:.2%}")
st.write(f"Expected Annual Volatility: {opt_volatility:.2%}")

if opt_style == "Maximize Sharpe Ratio":
    sharpe_ratio = (opt_return - rf) / opt_volatility
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")


fig, ax = plt.subplots()
weights_table['Weight'].plot(kind='bar', ax=ax)
ax.set_title("optimal portfolio Weights")
ax.set_ylabel("Weight")
st.pyplot(fig)


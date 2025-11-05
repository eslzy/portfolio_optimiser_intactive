import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from scipy.optimize import minimize

st.title("Portfolio Optimizer")

#stock inputs
with st.sidebar:
    if 'tickers' not in st.session_state:
        st.session_state['tickers'] = [] 

    new_ticker = st.text_input("Enter a ticker").upper()
                                                                #st.session_state is super useful, it keeps teh inputed data and prevents it from being resetted everytime you select other options, first time using it, its pretty good
    if st.button("Add stock") and new_ticker:
        if new_ticker not in st.session_state['tickers']:
            st.session_state['tickers'].append(new_ticker)
        else:
            st.warning("already in the portfolio.")

    enddate = dt.datetime.now()
    years = st.number_input("years analyzed:", min_value=1, value=5) #its buggy af and usually returns a yfrate limit reached err when you select 5 years 
    startdate = enddate - dt.timedelta(days=365 * years)

    st.write("Portfolio:")
    st.write(st.session_state['tickers'])

    rem = st.selectbox("Remove a stock from portfolio", [""] + st.session_state['tickers'])
    if st.button("Remove") and rem:
        st.session_state['tickers'].remove(rem)


#rfr selection
region = st.selectbox("Select Risk-Free Rate Region", ["US", "Europe", "Custom rate"]) #idk fi this is worth it especially for multi currency portfolios idk how well handle that 

if region == "US":
    rf_data = yf.download("^IRX", period="1mo")["Close"] # this is the best ticker i could find for the us rfr, might change later
    rf = float(rf_data.dropna().iloc[-1]) / 100  # had to convert to float or else it would return a dataframe and would lead to the sharpe ratio also being a dataframe
elif region == "Europe":
    rf_data = yf.download("^IRC", period="1mo")["Close"]  # European long-term rate
    rf = float(rf_data.dropna().iloc[-1]) / 100
elif region == "Custom rate":
    rf = st.number_input("Enter custom annual risk-free rate (in decimal)", 0.02)
else:
    rf = 0.02
    


#option selection, i want to be able to optimise for sharpe, for risk preferenc and for both. returning respectively the tangency portfolio, a portfolio on the efficient frontier and portfolios on the capital market line
#i'll handle them all separately, tried doing it all at once and it was super error prone

opt_styles = st.multiselect("Select optimization methods:", ["Maximize Sharpe Ratio", "Optimize for Risk Preference"], default=["Maximize Sharpe Ratio"])
if not len(st.session_state['tickers'])== 0:
    data = yf.download(st.session_state['tickers'], start=startdate, end=enddate)
else:
    st.write("Enter stocks in your portfolio")
    st.stop()
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

#-------sharpe and risk pref opt (CML)-------------------------------------------------------------------------------------------------------------------------------------------------------
if "Optimize for Risk Preference" in opt_styles and "Maximize Sharpe Ratio" in opt_styles:
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    res = minimize(neg_sharpe, start, args=(mean_returns, covariance, rf), method='SLSQP', bounds=bounds, constraints=constraints)
    
    opt_weights_tan = res.x         # here we calculate the tangency portfolio (same as the max_sharpe one) we'll consider this portfolio as one asset
    (port_return_tan, port_vol_tan) = portfolio_perf(opt_weights_tan, mean_returns, covariance)
    
    target_risk = st.slider("Select target volatility (risk %):", min_value=0.0, max_value=0.6, value=0.15, step=0.01)
    
    w_t = target_risk / port_vol_tan   #port_risk = w_t * tangency_port_vol
    w_rf = 1 - w_t  #weight for rf asset
    
    port_return = rf*(1 - w_t) + port_return_tan * w_t
    port_vol = target_risk    # here we assume the rf asset is truly risk free but close enough for us to say this 
    sharpe = (port_return - rf) / port_vol
    
    final_weights = opt_weights_tan * w_t   # weights for risky assets scaled by weight in tangency portfolio
    
    # Now include the rf asset weight in the DataFrame for display
    weights_df = pd.DataFrame({'Ticker': tickers + ['Risk-Free'], 'Weight (in total portfolio)': np.append(final_weights, w_rf)})
    
    st.subheader("Capital Market Line Optimal Portfolio (Risk Free + Tangency Portfolio). **must allow shorting**")
    st.write("Expected Annual Return:", port_return)
    st.write("Annual Volatility:", port_vol)
    st.write("Sharpe Ratio:", sharpe)
    st.write("Risk-Free Rate:", rf)

    
    st.bar_chart(weights_df.set_index('Ticker'))
    st.dataframe(weights_df.style.format({"Weight (in total portfolio)": "{:.2%}"}))
#------max sharpe opt--------------------------------------------------------------------------------------------------------------------------------------------------------
elif "Maximize Sharpe Ratio" in opt_styles and "Optimize for Risk Preference" not in opt_styles:
    
    #Optimize (Sharpe ratio)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1} #in section bc we have diff constraints for other selections
    res = minimize(neg_sharpe, start, args=(mean_returns, covariance, rf), method='SLSQP', bounds=bounds,constraints=constraints)

    opt_weights = res.x
    (port_return, port_vol) = portfolio_perf(opt_weights, mean_returns, covariance)
    sharpe = (port_return - rf) / port_vol

    #results
    st.subheader("Optimal Tangency Portfolio (Max Sharpe Ratio)")
    st.write("Expected Annual Return:", port_return)
    st.write("Annual Volatility:", port_vol)
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
    st.write("Annual Volatility:", port_vol)
    st.write("Sharpe Ratio:", sharpe)
    st.write("Risk Free Rate:",rf)

    # Create and display a bar chart of weights
    weights_df = pd.DataFrame({'Ticker': tickers,'Weight': opt_weights})
    st.bar_chart(weights_df.set_index('Ticker'))

    # Optional: show numeric table too
    st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}))
    
#sometimes this is calculated with the utility fuction so maybe ill see if its better to do tah tinstead of max teh return under the volatility constraint

#--------------------------------------------------------------------------------------------------------------------------------------------------------------


else:
    st.write("Select optimisation type")



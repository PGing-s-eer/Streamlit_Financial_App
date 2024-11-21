import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Configuration de la page
st.set_page_config(layout="wide")

# Initialize session state variables
if 'menu' not in st.session_state:
    st.session_state.menu = "Menu"
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'AAPL'
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = yf.Ticker(st.session_state.selected_ticker)
if 'financial_type' not in st.session_state:
    st.session_state.financial_type = 'Income Statement'
if 'period_type' not in st.session_state:
    st.session_state.period_type = 'Annual'


# Button of Navigation / Header
menu_tabs = ["Menu", "Chart", "Financial", "Monte Carlo", "Comparison","Portfolio"]
selected_tab = st.columns(len(menu_tabs))

for i, tab in enumerate(menu_tabs):
    with selected_tab[i]:
        if st.button(tab):
            st.session_state.menu = tab
menu = st.session_state.menu

# Initialisation des tickers du S&P 500
@st.cache_data
def list_wikipedia_sp500() -> pd.DataFrame:
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)[0]
    sp500_table.set_index('Symbol', inplace=True)  # Utiliser les tickers comme index
    return sp500_table

# Charger les tickers
if 'ticker_list' not in st.session_state:
    df_ticker = list_wikipedia_sp500()
    st.session_state.ticker_list = sorted(df_ticker.index.to_list())  # Liste triée

# Diviser la page en 3 colonnes
col_header1, col_header2, col_header3 = st.columns([2, 1, 1])

with col_header1:
    stock_data = st.session_state.get('stock_data', None)
    if stock_data:
        st.markdown(
            f"<h2 style='text-align: left; color: black;'>Selected Stock: "
            f"{stock_data.info.get('longName', 'N/A')}</h2>",
            unsafe_allow_html=True
        )

with col_header2:
    st.markdown(
        "<style>div.stButton > button {width: 100%; height: 50px; font-size: 18px;}</style>",
        unsafe_allow_html=True
    )
    if st.button("Upload Data") and 'selected_ticker' in st.session_state:
        st.session_state.stock_data = yf.Ticker(st.session_state.selected_ticker)

with col_header3:
    selected_ticker_temp = st.selectbox(
        "Select a Stock",
        st.session_state.ticker_list,
        index=st.session_state.ticker_list.index(st.session_state.selected_ticker)
        if 'selected_ticker' in st.session_state else 0
    )
    if selected_ticker_temp != st.session_state.get('selected_ticker', None):
        st.session_state.selected_ticker = selected_ticker_temp
        st.session_state.stock_data = yf.Ticker(selected_ticker_temp)

# Function to update date range based on the selected period 
def update_dates(period):
    end_date = datetime.now()
    if period == '1M':
        start_date = end_date - timedelta(days=30)
    elif period == '3M':
        start_date = end_date - timedelta(days=90)
    elif period == '6M':
        start_date = end_date - timedelta(days=180)
    elif period == 'YTD':
        start_date = datetime(end_date.year, 1, 1)
    elif period == '1Y':
        start_date = end_date - timedelta(days=365)
    elif period == '3Y':
        start_date = end_date - timedelta(days=3*365)
    elif period == '5Y':
        start_date = end_date - timedelta(days=5*365)
    elif period == 'MAX':
        hist_data = stock_data.history(period="max")
        start_date = hist_data.index.min().to_pydatetime() if not hist_data.empty else datetime(2022, 1, 1)
    return start_date, end_date

# Main Content based on selected menu from the 5 buttons in the header
if st.session_state.stock_data is not None:
    stock_data = st.session_state.stock_data

    # Menu Page
    if menu == "Menu":
        col_info, col_chart = st.columns([1, 2])        #Divide space in 2
        with col_info:
            col1, col2 = st.columns([1, 1])     #Divide the information part in two as well
            with col1:
                st.markdown("<hr>", unsafe_allow_html=True)         #Adding Bar and space to get a good layout
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.write(f"**Previous Close** : {stock_data.info.get('previousClose', 'N/A')}")                                                         #Adding Each information from yahoo finance in the left column
                st.write(f"**Open** : {stock_data.info.get('open', 'N/A')}")
                st.write(f"**Bid** : {stock_data.info.get('bid', 'N/A')}")
                st.write(f"**Ask** : {stock_data.info.get('ask', 'N/A')}")
                st.write(f"**Day's Range** : {stock_data.info.get('dayLow', 'N/A')} - {stock_data.info.get('dayHigh', 'N/A')}")
                st.write(f"**52 Week Range** : {stock_data.info.get('fiftyTwoWeekLow', 'N/A')} - {stock_data.info.get('fiftyTwoWeekHigh', 'N/A')}")
                st.write(f"**Volume** : {stock_data.info.get('volume', 'N/A'):,}")
                st.write(f"**Avg. Volume** : {stock_data.info.get('averageVolume', 'N/A'):,}")
                st.write(f"**Major Shareholders** : {stock_data.info.get('majorHoldersBreakdown', 'N/A')}")

            with col2:
                st.markdown("<hr>", unsafe_allow_html=True)                     #Adding Bar and space to get a good layout
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.write(f"**Market Cap** : {stock_data.info.get('marketCap', 'N/A'):,} USD")                                                           #Adding Each information from yahoo finance in the right column
                st.write(f"**PE Ratio (TTM)** : {stock_data.info.get('trailingPE', 'N/A')}")
                st.write(f"**EPS (TTM)** : {stock_data.info.get('trailingEps', 'N/A')}")
                st.write(f"**Dividend** : {stock_data.info.get('dividendYield', 'N/A')}")
                st.write(f"**Beta** : {stock_data.info.get('beta', 'N/A')}")
                st.write(f"**Earnings Date** : {stock_data.info.get('earningsDate', 'N/A')}")
                st.write(f"**Forward Dividend & Yield** : {stock_data.info.get('dividendYield', 'N/A')}")
                st.write(f"**Company Profile** : {stock_data.info.get('industry', 'N/A')} - {stock_data.info.get('sector', 'N/A')}")

        with col_chart:                 #2nd column of the menu page : Selectbox and chart 
                    period = st.selectbox("Select the period :", ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX'])        #Differents period availables
                    period_dict = {             #Dictionnary to match with '.history' function of yahoo finance afterwards
                        '1M': '1mo',
                        '3M': '3mo',
                        '6M': '6mo',
                        'YTD': 'ytd',
                        '1Y': '1y',
                        '3Y': '3y',
                        '5Y': '5y',
                        'MAX': 'max'
                    }
                    yf_period = period_dict.get(period, '1d')
                    hist = stock_data.history(period=yf_period)         #downloding the necessary data
                    fig = go.Figure(data=[go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close')])       #Displaying the chart       
                    fig.update_layout(title=f"Price of {st.session_state.selected_ticker} on {period}", xaxis_title="Date", yaxis_title="Prix (USD)")
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)             #Outside any column of the menu page : the description information from the selected ticker.
        st.subheader("Company Description")
        st.write(f"{stock_data.info.get('longBusinessSummary', 'N/A')}")

    # Chart Page
    if menu == "Chart":                 
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Chart of Stock Price")

        col_period, col_start_date, col_end_date = st.columns([1, 1, 1])            # Divide the page in 3 columns
        with col_period:                # Period Column
            period = st.selectbox("Select the period :", ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX'])
        start_date, end_date = update_dates(period)
        with col_start_date:            # Starting Date Column
            start_date = st.date_input("Starting Date", value=start_date)
        with col_end_date:              # Ending Date Column
            end_date = st.date_input("Ending Date", value=end_date)
                                        # Interval and type of plot Selection :
        interval = st.selectbox("Select an Interval :", ['1d', '1wk', '1mo'])
        chart_type = st.radio("Chart type :", ['Line plot', 'Candel plot'])
                                        # Downloading the needed data
        data = stock_data.history(start=start_date, end=end_date, interval=interval)
        data['MA50'] = data['Close'].rolling(window=50).mean()
        fig = go.Figure()

        if chart_type == 'Line plot':                                                                                       # Displaying based on selected options.
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
        else:
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candlestick'))
        
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='MA50', line=dict(color='purple')))
        colors = ['green' if row.Close > row.Open else 'red' for index, row in data.iterrows()]
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker=dict(color=colors, opacity=0.3), yaxis='y2'))
        fig.update_layout(
            title=f"{st.session_state.selected_ticker} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            yaxis2=dict(title="Volume", overlaying='y', side='right', showgrid=False, range=[0, max(data['Volume']) * 6]),
            height=700, xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # Financial Page
    if menu == "Financial":
        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])                      # Dividing the page in 5 columns of differents sizes
        with col1:                                                                      # Type of tab
            if st.button("Income Statement"):
                st.session_state.financial_type = 'Income Statement'
        with col2:
            if st.button("Balance Sheet"):
                st.session_state.financial_type = 'Balance Sheet'
        with col3:
            if st.button("Cash Flow"):
                st.session_state.financial_type = 'Cash Flow'
        with col4:                                                                      # Period 
            if st.button("Annual"):
                st.session_state.period_type = 'Annual'
        with col5:
            if st.button("Quarterly"):
                st.session_state.period_type = 'Quarterly'

        financial_type = st.session_state.financial_type
        period_type = st.session_state.period_type

        data_map = {                                                                    # Dis^playing
            ('Income Statement', 'Annual'): stock_data.financials,
            ('Income Statement', 'Quarterly'): stock_data.quarterly_financials,
            ('Balance Sheet', 'Annual'): stock_data.balance_sheet,
            ('Balance Sheet', 'Quarterly'): stock_data.quarterly_balance_sheet,
            ('Cash Flow', 'Annual'): stock_data.cashflow,
            ('Cash Flow', 'Quarterly'): stock_data.quarterly_cashflow
        }
        data = data_map.get((financial_type, period_type), None)

        st.subheader(f"{financial_type} - {period_type}")
        st.dataframe(data if data is not None else "No data available")





    # Page Monte Carlo Simulation
    if menu == "Monte Carlo":

        st.markdown("<hr>", unsafe_allow_html=True)

        # Options selections
        col_simulations, col_days, col_startprice, col_var = st.columns([1, 1, 1, 1])                           # Dividing in 4 columns 
        with col_simulations:                   # Column Nb of simulations with a selectbox
            num_simulations = st.selectbox("Select number of simulations (n):", [200, 500, 1000], index=1)      
        with col_days:                          # Column time hoizon with a selectbox
            time_horizon = st.selectbox("Select time horizon (days):", [30, 60, 90], index=0)

        # Downloading the data needed with 'history' function 
        stock_data = st.session_state.stock_data.history(period="1y")
        closing_prices = stock_data['Close']

        # Daily variation of return computing
        daily_returns = closing_prices.pct_change().dropna()
        mean_return = daily_returns.mean()
        std_dev = daily_returns.std()

        # Initialisation of the price
        last_price = closing_prices[-1]

        # Simulation of the price's trajectory
        simulation_results = []
        for _ in range(num_simulations):
            prices = [last_price]
            for _ in range(time_horizon):
                price_change = prices[-1] * (1 + np.random.normal(mean_return, std_dev))
                prices.append(price_change)
            simulation_results.append(prices)

        # Calculer la VaR à 95%  
        final_prices = [simulation[-1] for simulation in simulation_results]
        VaR_95 = np.percentile(final_prices, 5)

        with col_startprice:            # Column Starting price
            st.markdown("<vr>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size:20px; font-weight:bold;'>Starting price : ${last_price:.2f}</h4>", unsafe_allow_html=True)
        with col_var:                   # Column displaying the VaR
            st.markdown(f"<h4 style='font-size:20px; font-weight:bold;'>The Value at Risk (VaR) at 95% confidence interval is: ${VaR_95:.2f}</h4>", unsafe_allow_html=True)

        # Simulation Chart's display
        fig = go.Figure()
        for simulation in simulation_results:                   # Compacted code for simulations
            fig.add_trace(go.Scatter(x=list(range(time_horizon + 1)), y=simulation, mode='lines', line=dict(width=1), opacity=0.5))

        # Starting price line
        fig.add_trace(go.Scatter(x=[0], y=[last_price], mode='markers', marker=dict(color='red', size=8), name='Current Price'))
        fig.add_shape(
            type="line",
            x0=0,
            x1=time_horizon,
            y0=last_price,
            y1=last_price,
            line=dict(color="red", width=2, dash="dash"),
            name='Current Stock Price'
        )
            # Layout
        fig.update_layout(
            title=f"Monte Carlo Simulation for {st.session_state.selected_ticker} Stock Price in Next {time_horizon} Days",
            xaxis_title="Day",
            yaxis_title="Price (USD)",
            height=570,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    if menu == "Comparison":    # Comparison Page
       

        # Affichage dans les colonnes
        col_title, col_selectbox = st.columns([1, 1])               # Dividing the page in two columns
        with col_title:
            st.markdown(f"<h2 style='text-align: left; color: black;'>Compared Stock : </h2>", unsafe_allow_html=True)

        with col_selectbox:                                         # 2nd SelectBox for comparison
            selected_ticker_2 = st.selectbox(
                "Select a stock to compare",
                st.session_state.ticker_list,
                index=0  
            )

        # Initialisation of the stock_data_2 and the 2nd company's longName 
        stock_data_2 = yf.Ticker(selected_ticker_2)
        company_name_2 = stock_data_2.info.get('longName', 'N/A')

        # Period Initialisation
        if 'comparison_period' not in st.session_state:
            st.session_state.comparison_period = '1y'               # Défault Value
        st.markdown("<hr>", unsafe_allow_html=True)


        # Buttons for selecting the period for the comparison chart
        col_button1, col_button2, col_button3, col_button4, col_button5, col_button6, col_button7, col_button8= st.columns(8)
        with col_button1:
            if st.button("1M"):
                st.session_state.comparison_period = '1mo'
        with col_button2:
            if st.button("3M"):
                st.session_state.comparison_period = '3mo'
        with col_button3:
            if st.button("6M"):
                st.session_state.comparison_period = '6mo'
        with col_button4:
            if st.button("1YTD"):
                st.session_state.comparison_period = 'ytd'
        with col_button5:
            if st.button("1Y"):
                st.session_state.comparison_period = '1y'
        with col_button6:
            if st.button("3Y"):
                st.session_state.comparison_period = '3y'
        with col_button7:
            if st.button("5Y"):
                st.session_state.comparison_period = '5y'
        with col_button8:
            if st.button("MAX"):
                st.session_state.comparison_period = 'max'

        # Getting the data thorugh 'history' function with yahoo finance from stock_data variables
        data_1 = stock_data.history(period=st.session_state.comparison_period)
        data_2 = stock_data_2.history(period=st.session_state.comparison_period)


        # Daily performance computation
        returns_1 = data_1['Close'].pct_change().dropna()
        returns_2 = data_2['Close'].pct_change().dropna()

        # Computing volatility
        volatility_1 = returns_1.std() * (252 ** 0.5) * 100  # Annualised on 252 days of stock exchange
        volatility_2 = returns_2.std() * (252 ** 0.5) * 100  # Annualised on 252 days of stock exchange
       
        # Calculate average annual yield
        mean_return_1 = returns_1.mean() * 252 * 100  # Annualised on 252 days of stock exchange
        mean_return_2 = returns_2.mean() * 252 * 100  # Annualised on 252 days of stock exchange

        # bêta computing 
        beta_1 = stock_data.info.get('beta', 'N/A')
        beta_2 = stock_data_2.info.get('beta', 'N/A')

        # Sharpe Ratio 
        sharpe_ratio_1 = mean_return_1 / volatility_1
        sharpe_ratio_2 = mean_return_2 / volatility_2

        # Créer le graphique comparatif
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_1.index, y=data_1['Close'], mode='lines', name=f"{st.session_state.selected_ticker} Close"))
        fig.add_trace(go.Scatter(x=data_2.index, y=data_2['Close'], mode='lines', name=f"{selected_ticker_2} Close"))

        # Graph layout
        fig.update_layout(
            title="Comparison of Stock Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)


        st.markdown("<hr>", unsafe_allow_html=True)
        # Displaying Key statistics below the comparison chart
        st.markdown("### Key Statistics Comparison")

        # taking informations on both stock with '.info.get'
        key_stats_1 = {
            "Market Cap": stock_data.info.get('marketCap', 'N/A'),
            "PE Ratio (TTM)": stock_data.info.get('trailingPE', 'N/A'),
            "EPS (TTM)": stock_data.info.get('trailingEps', 'N/A'),
            "Dividend Yield": stock_data.info.get('dividendYield', 'N/A')
        }

        key_stats_2 = {
            "Market Cap": stock_data_2.info.get('marketCap', 'N/A'),
            "PE Ratio (TTM)": stock_data_2.info.get('trailingPE', 'N/A'),
            "EPS (TTM)": stock_data_2.info.get('trailingEps', 'N/A'),
            "Dividend Yield": stock_data_2.info.get('dividendYield', 'N/A')
        }

        # Displaying Statistics in two columns for a better comparison
        col_stat_1, col_stat_2 = st.columns(2)
        with col_stat_1:
            st.markdown(f"**{stock_data.info.get('longName', 'N/A')} - Key Statistics**")
            for stat, value in key_stats_1.items():
                st.write(f"{stat}: {value}")

        with col_stat_2:
            st.markdown(f"**{stock_data_2.info.get('longName', 'N/A')} - Key Statistics**")
            for stat, value in key_stats_2.items():
                st.write(f"{stat}: {value}")

        # Comparison of average yields over the selected period
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Average Return Comparison")
        col_ar1, col_ar2 = st.columns(2)
        with col_ar1:
            avg_return_1 = data_1['Close'].pct_change().mean() * 100
            st.write(f"Average Return for **{st.session_state.selected_ticker}** over the selected period: **{avg_return_1:.2f}%**")
        with col_ar2:
            avg_return_2 = data_2['Close'].pct_change().mean() * 100
            st.write(f"Average Return for **{selected_ticker_2}** over the selected period : **{avg_return_2:.2f}%**")

               
        # Performance indicator Part
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Performance Indicators")
        col_perf1, col_perf2 = st.columns(2)

        with col_perf1:
            st.markdown(f"**{st.session_state.selected_ticker}**")
            st.write(f"Volatility: {volatility_1:.2f}%")                            # Displaying the previous calculated results 
            st.write(f"Mean Return: {mean_return_1:.2f}%")
            st.write(f"Beta: {beta_1}")
            st.write(f"Sharpe Ratio: {sharpe_ratio_1:.2f}")

        with col_perf2:
            st.markdown(f"**{selected_ticker_2}**")
            st.write(f"Volatility: {volatility_2:.2f}%")                            # Displaying the previous calculated results for stock 2
            st.write(f"Mean Return: {mean_return_2:.2f}%")
            st.write(f"Beta: {beta_2}")
            st.write(f"Sharpe Ratio: {sharpe_ratio_2:.2f}")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"### Historical Annualized/Monthlyized Performance Comparison (Selected Period : {st.session_state.comparison_period})")

        # Load data from both stocks for annualized performance
        data_full_1 = stock_data.history(period="max")
        data_full_2 = stock_data_2.history(period="max")

        # Calculate annualized performance for fixed periods (1, 3, 5, 10 years and Max)
        def calculate_cagr(data, num_years):
            if len(data) > 0:
                end_price = data['Close'].iloc[-1]
                start_price = data['Close'].iloc[0]
                cagr = ((end_price / start_price) ** (1 / num_years) - 1) * 100
                return cagr
            else:
                return None

       # Periods to be compared (in years)
        periods = [1, 3, 5, 10]

        performance_dict_1 = {}
        performance_dict_2 = {}

        for period in periods:
            num_days = period * 365  # Approximate number of days
            if len(data_full_1) >= num_days:
                performance_dict_1[f"{period}Y"] = calculate_cagr(data_full_1.tail(num_days), period)
            else:
                performance_dict_1[f"{period}Y"] = "N/A"

            if len(data_full_2) >= num_days:
                performance_dict_2[f"{period}Y"] = calculate_cagr(data_full_2.tail(num_days), period)
            else:
                performance_dict_2[f"{period}Y"] = "N/A"

        # Calculation of annualized performance for the “Max” period
        total_days_1 = (data_full_1.index[-1] - data_full_1.index[0]).days
        total_days_2 = (data_full_2.index[-1] - data_full_2.index[0]).days
        total_years_1 = total_days_1 / 365
        total_years_2 = total_days_2 / 365

        if total_years_1 >= 1:
            performance_dict_1["Max"] = calculate_cagr(data_full_1, total_years_1)
        else:
            performance_dict_1["Max"] = "N/A"

        if total_years_2 >= 1:
            performance_dict_2["Max"] = calculate_cagr(data_full_2, total_years_2)
        else:
            performance_dict_2["Max"] = "N/A"

        # Displaying the Historical Annualized Performance table
        st.markdown("#### Historical Annualized Performance (CAGR)")
        st.table(pd.DataFrame({
            'Period': performance_dict_1.keys(),
            f"{st.session_state.selected_ticker}": [f"{value:.2f}%" if isinstance(value, (int, float)) else value for value in performance_dict_1.values()],
            f"{selected_ticker_2}": [f"{value:.2f}%" if isinstance(value, (int, float)) else value for value in performance_dict_2.values()]
        }).set_index('Period'))

    if menu == "Portfolio": 
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Step 1 : Select the number of stocks in the portfolio
        st.subheader("Step 1: Select the number of stocks in your portfolio")
        num_stocks = st.slider("Number of stocks", min_value=1, max_value=7, value=3) #Min1 Max7 Default3

        col_which_stock, col_amount, col_value, col_percentage = st.columns([2, 2, 1, 1]) #Displaying 4 side to side columns for the usage
        with col_which_stock:
            # Step 2 : Select which stock
            st.subheader("Step 2: Select the stocks for your portfolio")
            selected_stocks = []            #Initialisation of the list
            for i in range(num_stocks):
                stock = st.selectbox(f"Select Stock {i + 1}", st.session_state.ticker_list, key=f"stock_{i}")
                selected_stocks.append(stock) #Adding the selected stock to the list


        with col_amount:
            #Step 3 :Defined the number of stock in the porfolio for each of them
            st.subheader("Step 3: Define the number of shares")

            # Initialisation of the rest
            quantities = []
            stock_values = []
            total_portfolio_value = 0

            # Get the actual prices for the selected stocks
            prices = {} #Initialisation of the dictionary
            for stock in selected_stocks:                           #for loop to get the price of each stock
                try:
                    data = yf.Ticker(stock).history(period="1d")
                    prices[stock] = data['Close'].iloc[-1]
                except:
                    st.error(f"Error for {stock}")
                    prices[stock] = 0

            # Creating slider to defined quantities
            for stock in selected_stocks:                                       #for loop to get the total value of each stock in the portfolio
                quantity = st.number_input(f"Number of shares for {stock}", min_value=0, value=10, key=f"quantity_{stock}")
                quantities.append(quantity)
                stock_value = quantity * prices[stock]
                stock_values.append(stock_value)
                total_portfolio_value += stock_value

        with col_value:
            st.subheader("**Stock Value ($)**")
            st.write("")
            for stock, stock_value in zip(selected_stocks, stock_values):   #Displaying the name and total value of each stock in the portfolio
                st.write(f"{stock}")
                st.write(f"${stock_value:,.2f}")

            st.markdown("<hr>", unsafe_allow_html=True)

        with col_percentage:
            st.subheader("**Allocation (%)**")
            st.write("")
            for stock, value in zip(selected_stocks, stock_values):         #Displaying the name and percentage of each stock in the portfolio
                st.write(f"{stock}")
                allocation = (value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
                st.write(f"{allocation:.2f}%")
            st.markdown("<hr>", unsafe_allow_html=True)

        #Display the value of the Portfolio in Total
        st.write(f"**Total Portfolio Value:** ${total_portfolio_value:,.2f}")       





        # PieChart Part :
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Portfolio Allocation Pie Chart")
        #Create the Fig
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=selected_stocks,  # Stock a=name
                    values=stock_values,  # Amount
                    hole=0.3,  # Donuts style
                    hoverinfo="label+percent+value"  # Hover infos
                )
            ]
        )
        # Display the Pie Chart
        st.plotly_chart(fig, use_container_width=True)





        #Porfolio Performance Part :
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Portfolio Performance Over Time")
        #Selection of the period with through buttons
        col_1y, col_5y, col_10y = st.columns(3)

        with col_1y:
            if st.button("1 Year"):
                selected_period = "1y"
        with col_5y:
            if st.button("5 Years"):
                selected_period = "5y"
        with col_10y:
            if st.button("10 Years"):
                selected_period = "10y"                                    
        if 'selected_period' not in locals():       # Default period 
            selected_period = "10y"

        # Take historical data of the selected stock + s&p500 using .history()
        historical_prices = {}
        sp500_data = None
        for stock in selected_stocks:
            # Selected stocks
            data = yf.Ticker(stock).history(period=selected_period)
            historical_prices[stock] = data['Close']
            # S&P 500 data
        sp500_data = yf.Ticker("^GSPC").history(period=selected_period)['Close']
        

        # Calculatation of the weighted portfolio performance
        if historical_prices and sp500_data is not None:
            portfolio_performance = None

            # Selected stocks data combination in one DF
            combined_data = pd.DataFrame(historical_prices)
            # Normalisation of the price to calculate performance
            normalized_data = combined_data / combined_data.iloc[0]
            sp500_normalized = sp500_data / sp500_data.iloc[0]
            # Apply the weights of each stock
            weights = [quantity / sum(quantities) for quantity in quantities]
            weighted_data = normalized_data.mul(weights, axis=1)
            # Sum to get the global performance
            portfolio_performance = weighted_data.sum(axis=1)


            # Line Chart Part
            # Porfolio Line
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=portfolio_performance.index,
                    y=portfolio_performance.values,
                    mode="lines",
                    name="Portfolio",
                    line=dict(width=4, color='blue',dash='dot')  # Ticker and dot the portfolio line
                )
            )

            # S&P500 Line
            fig.add_trace(
                go.Scatter(
                    x=sp500_normalized.index,
                    y=sp500_normalized.values,
                    mode="lines",
                    name="S&P 500",

                    line=dict(width=4, color='red', dash='dot')  # Ticker and dot the s&p500 line
                )
            )

            # Stocks Lines
            for stock in selected_stocks:
                fig.add_trace(
                    go.Scatter(
                        x=normalized_data.index,
                        y=normalized_data[stock],
                        mode="lines",
                        name=f"{stock}"
                    )
                )

            # Layout and Display Chart
            fig.update_layout(
                title=f"Portfolio vs. S&P 500 Performance Over {selected_period.upper()}",
                xaxis_title="Date",
                yaxis_title="Normalized Value (Base 100)",
                legend_title="Legend",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)




            # Porfolio Performance Stats Part
            portfolio_total_return = portfolio_performance[-1] / portfolio_performance[0] - 1
            sp500_total_return = sp500_data[-1] / sp500_data[0] - 1
            years = {"1y": 1, "5y": 5, "10y": 10}[selected_period]

            # Portfolio Annualized Performances
            portfolio_annualized_return = ((1 + portfolio_total_return) ** (1 / years) - 1) * 100
            sp500_annualized_return = ((1 + sp500_total_return) ** (1 / years) - 1) * 100

            # Portfolio Volatility Performance
            portfolio_daily_returns = portfolio_performance.pct_change().dropna()
            portfolio_annualized_volatility = portfolio_daily_returns.std() * (252 ** 0.5) * 100

            sp500_daily_returns = sp500_data.pct_change().dropna()
            sp500_annualized_volatility = sp500_daily_returns.std() * (252 ** 0.5) * 100

            # risk_free_rate assumption at 1.5
            risk_free_rate = 1.5

            # Sharpe Ratio
            portfolio_sharpe_ratio = (portfolio_annualized_return - risk_free_rate) / portfolio_annualized_volatility
            sp500_sharpe_ratio = (sp500_annualized_return - risk_free_rate) / sp500_annualized_volatility

            # Mean Return
            portfolio_mean_return = portfolio_daily_returns.mean() * 252 * 100
            sp500_mean_return = sp500_daily_returns.mean() * 252 * 100

            # Displaying the Dataframe
            stats = {
                "Metric": ["Annualized Return (%)", "Annualized Volatility (%)", "Sharpe Ratio", "Mean Return (%)"],
                "Portfolio": [
                    f"{portfolio_annualized_return:.2f}%",
                    f"{portfolio_annualized_volatility:.2f}%",
                    f"{portfolio_sharpe_ratio:.2f}",
                    f"{portfolio_mean_return:.2f}%"
                ],
                "S&P 500": [
                    f"{sp500_annualized_return:.2f}%",
                    f"{sp500_annualized_volatility:.2f}%",
                    f"{sp500_sharpe_ratio:.2f}",
                    f"{sp500_mean_return:.2f}%"
                ]
            }

            stats_df = pd.DataFrame(stats)
            st.subheader("Portfolio vs. S&P 500 Performance Metrics")
            st.table(stats_df)
        else:
            st.warning("No data available to plot the portfolio performance. Please adjust your selections.")



        




else:
    st.write("Please upload the data to get display the page")

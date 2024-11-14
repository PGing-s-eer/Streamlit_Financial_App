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

# Navigation Header
menu_tabs = ["Menu", "Chart", "Financial", "Monte Carlo", "Forecast", "Comparison"]
selected_tab = st.columns(len(menu_tabs))

for i, tab in enumerate(menu_tabs):
    with selected_tab[i]:
        if st.button(tab):
            st.session_state.menu = tab

menu = st.session_state.menu

# Header with Ticker Selection and Upload Button
col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
with col_header1:
    stock_data = st.session_state.stock_data
    st.markdown(f"<h2 style='text-align: left; color: black;'>Selected Stock : {stock_data.info.get('longName', 'N/A')}</h2>", unsafe_allow_html=True)

with col_header2:
    st.markdown("<style>div.stButton > button {width: 100%; height: 50px; font-size: 18px;}</style>", unsafe_allow_html=True)
    if st.button("Upload Data"):
        st.session_state.stock_data = yf.Ticker(st.session_state.selected_ticker)

with col_header3:
    selected_ticker_temp = st.selectbox(
        "Sélectionnez une action",
        ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        index=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'].index(st.session_state.selected_ticker),
    )
    if selected_ticker_temp != st.session_state.selected_ticker:
        st.session_state.selected_ticker = selected_ticker_temp

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

# Main Content based on selected menu
if st.session_state.stock_data is not None:
    stock_data = st.session_state.stock_data

    # Menu Page
    if menu == "Menu":
        col_info, col_chart = st.columns([1, 2])
        with col_info:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.write(f"**Previous Close** : {stock_data.info.get('previousClose', 'N/A')}")
                st.write(f"**Open** : {stock_data.info.get('open', 'N/A')}")
                st.write(f"**Bid** : {stock_data.info.get('bid', 'N/A')}")
                st.write(f"**Ask** : {stock_data.info.get('ask', 'N/A')}")
                st.write(f"**Day's Range** : {stock_data.info.get('dayLow', 'N/A')} - {stock_data.info.get('dayHigh', 'N/A')}")
                st.write(f"**52 Week Range** : {stock_data.info.get('fiftyTwoWeekLow', 'N/A')} - {stock_data.info.get('fiftyTwoWeekHigh', 'N/A')}")
                st.write(f"**Volume** : {stock_data.info.get('volume', 'N/A'):,}")
                st.write(f"**Avg. Volume** : {stock_data.info.get('averageVolume', 'N/A'):,}")
                st.write(f"**Major Shareholders** : {stock_data.info.get('majorHoldersBreakdown', 'N/A')}")

            with col2:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.write(f"**Market Cap** : {stock_data.info.get('marketCap', 'N/A'):,} USD")
                st.write(f"**PE Ratio (TTM)** : {stock_data.info.get('trailingPE', 'N/A')}")
                st.write(f"**EPS (TTM)** : {stock_data.info.get('trailingEps', 'N/A')}")
                st.write(f"**Dividend** : {stock_data.info.get('dividendYield', 'N/A')}")
                st.write(f"**Beta** : {stock_data.info.get('beta', 'N/A')}")
                st.write(f"**Earnings Date** : {stock_data.info.get('earningsDate', 'N/A')}")
                st.write(f"**Forward Dividend & Yield** : {stock_data.info.get('dividendYield', 'N/A')}")
                st.write(f"**Company Profile** : {stock_data.info.get('industry', 'N/A')} - {stock_data.info.get('sector', 'N/A')}")

        with col_chart:
                    period = st.selectbox("Sélectionnez la période :", ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX'])
                    period_dict = {
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
                    hist = stock_data.history(period=yf_period)
                    fig = go.Figure(data=[go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close')])
                    fig.update_layout(title=f"Cours de {st.session_state.selected_ticker} sur {period}", xaxis_title="Date", yaxis_title="Prix (USD)")
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Company Description")
        st.write(f"{stock_data.info.get('longBusinessSummary', 'N/A')}")

    # Chart Page
    if menu == "Chart":
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Chart of Stock Price")

        col_period, col_start_date, col_end_date = st.columns([1, 1, 1])
        with col_period:
            period = st.selectbox("Sélectionnez la période :", ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX'])
        start_date, end_date = update_dates(period)
        with col_start_date:
            start_date = st.date_input("Date de début", value=start_date)
        with col_end_date:
            end_date = st.date_input("Date de fin", value=end_date)

        interval = st.selectbox("Sélectionnez l'intervalle de temps :", ['1d', '1wk', '1mo'])
        chart_type = st.radio("Type de graphique :", ['Ligne', 'Chandelier'])

        data = stock_data.history(start=start_date, end=end_date, interval=interval)
        data['MA50'] = data['Close'].rolling(window=50).mean()
        fig = go.Figure()

        if chart_type == 'Ligne':
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
        else:
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candlestick'))
        
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='MA50', line=dict(color='purple')))
        colors = ['green' if row.Close > row.Open else 'red' for index, row in data.iterrows()]
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker=dict(color=colors, opacity=0.3), yaxis='y2'))
        fig.update_layout(
            title=f"{st.session_state.selected_ticker} Stock Price",
            xaxis_title="Date",
            yaxis_title="Prix (USD)",
            yaxis2=dict(title="Volume", overlaying='y', side='right', showgrid=False, range=[0, max(data['Volume']) * 6]),
            height=700, xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # Financial Page
    if menu == "Financial":
        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
        with col1:
            if st.button("Income Statement"):
                st.session_state.financial_type = 'Income Statement'
        with col2:
            if st.button("Balance Sheet"):
                st.session_state.financial_type = 'Balance Sheet'
        with col3:
            if st.button("Cash Flow"):
                st.session_state.financial_type = 'Cash Flow'
        with col4:
            if st.button("Annual"):
                st.session_state.period_type = 'Annual'
        with col5:
            if st.button("Quarterly"):
                st.session_state.period_type = 'Quarterly'

        financial_type = st.session_state.financial_type
        period_type = st.session_state.period_type

        data_map = {
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

        # Sélecteurs pour le nombre de simulations et l'horizon temporel
        col_simulations, col_days, col_startprice, col_var = st.columns([1, 1, 1, 1])
        with col_simulations:
            num_simulations = st.selectbox("Select number of simulations (n):", [200, 500, 1000], index=1)
        with col_days:
            time_horizon = st.selectbox("Select time horizon (days):", [30, 60, 90], index=0)

        # Récupérer les données de l'action sélectionnée
        stock_data = st.session_state.stock_data.history(period="1y")
        closing_prices = stock_data['Close']

        # Calculer la variation journalière des rendements
        daily_returns = closing_prices.pct_change().dropna()
        mean_return = daily_returns.mean()
        std_dev = daily_returns.std()

        # Initialiser le prix actuel
        last_price = closing_prices[-1]

        # Simuler les trajectoires de prix
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

        with col_startprice:
            st.markdown("<vr>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size:20px; font-weight:bold;'>Starting price : ${last_price:.2f}</h4>", unsafe_allow_html=True)
        with col_var:
            # Display the starting price and VaR for the simulation
            st.markdown(f"<h4 style='font-size:20px; font-weight:bold;'>The Value at Risk (VaR) at 95% confidence interval is: ${VaR_95:.2f}</h4>", unsafe_allow_html=True)

        # Affichage du graphique des simulations
        fig = go.Figure()
        for simulation in simulation_results:
            fig.add_trace(go.Scatter(x=list(range(time_horizon + 1)), y=simulation, mode='lines', line=dict(width=1), opacity=0.5))

        # Tracer le prix actuel sur le graphique
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

        fig.update_layout(
            title=f"Monte Carlo Simulation for {st.session_state.selected_ticker} Stock Price in Next {time_horizon} Days",
            xaxis_title="Day",
            yaxis_title="Price (USD)",
            height=570,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    if menu == "Comparison":
        # Créer une deuxième selectbox pour l'action à compare
                # Initialiser la variable avec une valeur par défaut avant l'affichage

        # Affichage dans les colonnes
        col_title, col_selectbox = st.columns([1, 1])
        with col_title:
            st.markdown(f"<h2 style='text-align: left; color: black;'>Compared Stock : </h2>", unsafe_allow_html=True)

        with col_selectbox:
            selected_ticker_2 = st.selectbox(
                "Select a stock to compare",
                ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                index=0  # Ajuster l'index par défaut si nécessaire
            )

        # Mettre à jour la variable `company_name_2` après la sélection
        stock_data_2 = yf.Ticker(selected_ticker_2)
        company_name_2 = stock_data_2.info.get('longName', 'N/A')

        # Initialiser la période par défaut dans `st.session_state` si elle n'existe pas
        if 'comparison_period' not in st.session_state:
            st.session_state.comparison_period = '1y'  # Valeur par défaut
        st.markdown("<hr>", unsafe_allow_html=True)
        # Ajouter des boutons pour changer la période du graphique
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

        # Charger les données des deux actions avec la période sélectionnée
        data_1 = stock_data.history(period=st.session_state.comparison_period)
        data_2 = stock_data_2.history(period=st.session_state.comparison_period)


        # Calculer les rendements quotidiens
        returns_1 = data_1['Close'].pct_change().dropna()
        returns_2 = data_2['Close'].pct_change().dropna()

        # Calculer la volatilité (écart-type des rendements)
        volatility_1 = returns_1.std() * (252 ** 0.5) * 100  # Annualisée
        volatility_2 = returns_2.std() * (252 ** 0.5) * 100  # Annualisée

        # Calculer le rendement moyen annuel
        mean_return_1 = returns_1.mean() * 252 * 100  # Annualisé
        mean_return_2 = returns_2.mean() * 252 * 100  # Annualisé

        # Calculer la bêta (approximative, nécessite un indice de référence)
        beta_1 = stock_data.info.get('beta', 'N/A')
        beta_2 = stock_data_2.info.get('beta', 'N/A')

        # Ratio de Sharpe (rendement ajusté au risque)
        risk_free_rate = 1.5  # Taux sans risque hypothétique en pourcentage
        sharpe_ratio_1 = (mean_return_1 - risk_free_rate) / volatility_1
        sharpe_ratio_2 = (mean_return_2 - risk_free_rate) / volatility_2

        # Créer le graphique comparatif
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_1.index, y=data_1['Close'], mode='lines', name=f"{st.session_state.selected_ticker} Close"))
        fig.add_trace(go.Scatter(x=data_2.index, y=data_2['Close'], mode='lines', name=f"{selected_ticker_2} Close"))

        # Mise en page du graphique
        fig.update_layout(
            title="Comparison of Stock Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        # Afficher des statistiques clés des actions comparées sous le graphique
        st.markdown("### Key Statistics Comparison")

        # Obtenir les données financières clés des deux actions
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

        # Afficher les statistiques sous forme de tableau
        col_stat_1, col_stat_2 = st.columns(2)
        with col_stat_1:
            st.markdown(f"**{stock_data.info.get('longName', 'N/A')} - Key Statistics**")
            for stat, value in key_stats_1.items():
                st.write(f"{stat}: {value}")

        with col_stat_2:
            st.markdown(f"**{stock_data_2.info.get('longName', 'N/A')} - Key Statistics**")
            for stat, value in key_stats_2.items():
                st.write(f"{stat}: {value}")

        # Comparaison des rendements moyens sur la période sélectionnée
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Average Return Comparison")
        col_ar1, col_ar2 = st.columns(2)
        with col_ar1:
            avg_return_1 = data_1['Close'].pct_change().mean() * 100
            st.write(f"Average Return for **{st.session_state.selected_ticker}** over the selected period: **{avg_return_1:.2f}%**")
        with col_ar2:
            avg_return_2 = data_2['Close'].pct_change().mean() * 100
            st.write(f"Average Return for **{selected_ticker_2}** over the selected period : **{avg_return_2:.2f}%**")

        # Affichage des résultats
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Performance Indicators")
        col_perf1, col_perf2 = st.columns(2)

        with col_perf1:
            st.markdown(f"**{st.session_state.selected_ticker}**")
            st.write(f"Volatility: {volatility_1:.2f}%")
            st.write(f"Mean Return: {mean_return_1:.2f}%")
            st.write(f"Beta: {beta_1}")
            st.write(f"Sharpe Ratio: {sharpe_ratio_1:.2f}")

        with col_perf2:
            st.markdown(f"**{selected_ticker_2}**")
            st.write(f"Volatility: {volatility_2:.2f}%")
            st.write(f"Mean Return: {mean_return_2:.2f}%")
            st.write(f"Beta: {beta_2}")
            st.write(f"Sharpe Ratio: {sharpe_ratio_2:.2f}")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"### Historical Annualized/Monthlyized Performance Comparison (Selected Period : {st.session_state.comparison_period})")

        # Charger les données des deux stocks pour les performances annualisées
        data_full_1 = stock_data.history(period="max")
        data_full_2 = stock_data_2.history(period="max")

        # Calculer la performance annualisée pour des périodes fixes (1, 3, 5, 10 ans et Max)
        def calculate_cagr(data, num_years):
            if len(data) > 0:
                end_price = data['Close'].iloc[-1]
                start_price = data['Close'].iloc[0]
                cagr = ((end_price / start_price) ** (1 / num_years) - 1) * 100
                return cagr
            else:
                return None

        # Périodes à comparer (en années)
        periods = [1, 3, 5, 10]

        performance_dict_1 = {}
        performance_dict_2 = {}

        for period in periods:
            num_days = period * 365  # Approximation du nombre de jours
            if len(data_full_1) >= num_days:
                performance_dict_1[f"{period}Y"] = calculate_cagr(data_full_1.tail(num_days), period)
            else:
                performance_dict_1[f"{period}Y"] = "N/A"

            if len(data_full_2) >= num_days:
                performance_dict_2[f"{period}Y"] = calculate_cagr(data_full_2.tail(num_days), period)
            else:
                performance_dict_2[f"{period}Y"] = "N/A"

        # Calcul de la performance annualisée pour la période "Max"
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

        # Affichage du tableau de performances
        st.markdown("#### Historical Annualized Performance (CAGR)")
        st.table(pd.DataFrame({
            'Period': performance_dict_1.keys(),
            f"{st.session_state.selected_ticker}": [f"{value:.2f}%" if isinstance(value, (int, float)) else value for value in performance_dict_1.values()],
            f"{selected_ticker_2}": [f"{value:.2f}%" if isinstance(value, (int, float)) else value for value in performance_dict_2.values()]
        }).set_index('Period'))

        # Charger les données pour le graphique en fonction du bouton de période cliqué
        data_1 = stock_data.history(period=st.session_state.comparison_period)
        data_2 = stock_data_2.history(period=st.session_state.comparison_period)

        # Créer le graphique comparatif
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_1.index, y=data_1['Close'], mode='lines', name=f"{st.session_state.selected_ticker} Close"))
        fig.add_trace(go.Scatter(x=data_2.index, y=data_2['Close'], mode='lines', name=f"{selected_ticker_2} Close"))

        # Mise en page du graphique
        fig.update_layout(
            title="Comparison of Stock Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600
        )

 


else:
    st.write("Veuillez télécharger les données pour voir les informations de l'action.")

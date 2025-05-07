import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
import plotly.graph_objects as go
from PIL import Image

# Streamlit layout
st.set_page_config(layout="wide")

# Sidebar setup
st.sidebar.title("Binance Data")
#image_sidebar = Image.open("Pic1.png")  # Sidebar image
#st.sidebar.image(image_sidebar, use_column_width=True)

# Secure API Key inputs
API_KEY = st.sidebar.text_input("Binance API Key", type="password")
API_SECRET = st.sidebar.text_input("Binance API Secret", type="password")

# Trading pair inputs
symbol = st.sidebar.text_input("Symbol", "BTCUSDT")
interval = st.sidebar.selectbox("Interval", options=["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
lookback = st.sidebar.text_input("Lookback", "1 day ago UTC")

# Main page image
#image_main = Image.open("Pic2.png")
#st.image(image_main, use_column_width=True)

# Remove extra space at the top
st.markdown("<style> .css-18e3th9 { padding-top: 0; } </style>", unsafe_allow_html=True)

# Function to fetch Binance candlestick data
def get_historical_klines(api_key, api_secret, symbol, interval, lookback):
    try:
        client = Client(api_key, api_secret)
        klines = client.get_historical_klines(symbol, interval, lookback)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        return df
    except Exception as e:
        raise Exception(f"Error fetching data: {e}")

# Function to calculate EMAs
def add_ema(df, periods=[20, 50, 100, 200]):
    for period in periods:
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return df

# Function to plot chart
def plot_data_with_ema(df):
    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlesticks'
    ))

    # EMAs
    for ema_period in [20, 50, 100, 200]:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[f'EMA_{ema_period}'],
            mode='lines',
            name=f'EMA {ema_period}'
        ))

    fig.update_layout(
        title="Candlestick Chart with EMAs",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

# Fetch and display data
if API_KEY and API_SECRET:
    try:
        df = get_historical_klines(API_KEY, API_SECRET, symbol, interval, lookback)
        df = add_ema(df)

        st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>Binance API Analysis</h1>", unsafe_allow_html=True)

        # Current values
        current_price = df['close'].iloc[-1]
        ema_20 = df['EMA_20'].iloc[-1]
        ema_50 = df['EMA_50'].iloc[-1]
        ema_100 = df['EMA_100'].iloc[-1]
        ema_200 = df['EMA_200'].iloc[-1]

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(
                f"""
                <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; text-align: center;">
                    <h3>Current Price</h3>
                    <p style="font-size: 24px; font-weight: bold;">${current_price:,.6f}</p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center;">
                    <h3>EMA 20</h3>
                    <p style="font-size: 24px; font-weight: bold;">${ema_20:,.6f}</p>
                </div>
                """, unsafe_allow_html=True)

        with col3:
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center;">
                    <h3>EMA 50</h3>
                    <p style="font-size: 24px; font-weight: bold;">${ema_50:,.6f}</p>
                </div>
                """, unsafe_allow_html=True)

        with col4:
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center;">
                    <h3>EMA 100</h3>
                    <p style="font-size: 24px; font-weight: bold;">${ema_100:,.6f}</p>
                </div>
                """, unsafe_allow_html=True)

        with col5:
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center;">
                    <h3>EMA 200</h3>
                    <p style="font-size: 24px; font-weight: bold;">${ema_200:,.6f}</p>
                </div>
                """, unsafe_allow_html=True)

        # Plot
        plot_data_with_ema(df)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.warning("Please enter your Binance API Key and Secret in the sidebar.")

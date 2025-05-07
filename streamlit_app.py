import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Sidebar - API credentials
st.sidebar.title("Bybit API Credentials")
api_key = st.sidebar.text_input("API Key", type="password")
api_secret = st.sidebar.text_input("API Secret", type="password")

# Sidebar - Trading parameters
st.sidebar.title("Trading Parameters")
symbol = st.sidebar.text_input("Symbol", "BTCUSDT").upper()
interval = st.sidebar.selectbox("Interval", options=["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"], index=5)
lookback_days = st.sidebar.number_input("Lookback Period (days)", min_value=1, max_value=365, value=1)

# Function to fetch historical klines from Bybit
def get_historical_klines(symbol, interval, lookback_days):
    """
    Fetch historical klines (candlestick) data from Bybit.

    :param symbol: Trading pair symbol (e.g., 'BTCUSDT')
    :param interval: Timeframe for candlesticks (e.g., '1', '5', '15', '60', 'D')
    :param lookback_days: Number of days to look back
    :return: Pandas DataFrame with OHLCV data
    """
    base_url = "https://api.bybit.com"
    endpoint = "/v5/market/kline"
    url = base_url + endpoint

    end_time = int(time.time() * 1000)
    start_time = end_time - (lookback_days * 24 * 60 * 60 * 1000)

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "start": start_time,
        "end": end_time,
        "limit": 1000
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data["retCode"] != 0:
            raise Exception(f"API Error: {data['retMsg']}")

        df = pd.DataFrame(data["result"]["list"], columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)

        return df
    except Exception as e:
        raise Exception(f"Error fetching data: {e}")

# Function to add EMAs to the DataFrame
def add_ema(df, periods=[20, 50, 100, 200]):
    """
    Add Exponential Moving Averages (EMAs) to the DataFrame.

    :param df: DataFrame with price data
    :param periods: List of periods for EMAs
    :return: DataFrame with added EMA columns
    """
    for period in periods:
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return df

# Function to plot candlestick chart with EMAs
def plot_data_with_ema(df):
    """
    Create an interactive Plotly plot with candlestick data and EMAs.

    :param df: DataFrame with price and EMA data
    """
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlesticks'
    ))

    # Add EMAs
    for ema_period in [20, 50, 100, 200]:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[f'EMA_{ema_period}'],
            mode='lines',
            name=f'EMA {ema_period}'
        ))

    # Customize layout
    fig.update_layout(
        title=f"{symbol} Candlestick Chart with EMAs",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

# Main application
st.title("Bybit Candlestick Chart with EMAs")

# Fetch and process data
try:
    df = get_historical_klines(symbol, interval, lookback_days)
    df = add_ema(df)

    # Display latest price and EMAs
    current_price = df['close'].iloc[-1]
    ema_20 = df['EMA_20'].iloc[-1]
    ema_50 = df['EMA_50'].iloc[-1]
    ema_100 = df['EMA_100'].iloc[-1]
    ema_200 = df['EMA_200'].iloc[-1]

    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(label="Current Price", value=f"${current_price:,.2f}")

    with col2:
        st.metric(label="EMA 20", value=f"${ema_20:,.2f}")

    with col3:
        st.metric(label="EMA 50", value=f"${ema_50:,.2f}")

    with col4:
        st.metric(label="EMA 100", value=f"${ema_100:,.2f}")

    with col5:
        st.metric(label="EMA 200", value=f"${ema_200:,.2f}")

    # Plot chart
    plot_data_with_ema(df)

except Exception as e:
    st.error(f"Error: {e}")

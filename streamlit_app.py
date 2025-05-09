import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
import plotly.graph_objects as go
from PIL import Image

# Initialize Binance client without API keys (public data only)
client = Client()

def get_historical_klines(symbol, interval, lookback):
    try:
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

def add_ema(df, periods=[20, 50, 100, 200]):
    for period in periods:
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return df

def plot_data_with_ema(df):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlesticks'
    ))

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

# Streamlit layout
st.set_page_config(layout="wide")
st.sidebar.title("Binance Data")

# Optional image display (can comment out if not available)
try:
    image_sidebar = Image.open("Pic1.png")
    st.sidebar.image(image_sidebar, use_column_width=True)
except:
    pass

symbol = st.sidebar.text_input("Symbol", "BTCUSDT")
interval = st.sidebar.selectbox("Interval", options=["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
lookback = st.sidebar.text_input("Lookback", "1 day ago UTC")

# Optional main image
try:
    image_main = Image.open("Pic2.png")
    st.image(image_main, use_column_width=True)
except:
    pass

# Remove top padding
st.markdown("<style> .css-18e3th9 { padding-top: 0; } </style>", unsafe_allow_html=True)

try:
    df = get_historical_klines(symbol, interval, lookback)
    df = add_ema(df)

    st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>Binance API Analysis</h1>", unsafe_allow_html=True)

    current_price = df['close'].iloc[-1]
    ema_20 = df['EMA_20'].iloc[-1]
    ema_50 = df['EMA_50'].iloc[-1]
    ema_100 = df['EMA_100'].iloc[-1]
    ema_200 = df['EMA_200'].iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.markdown(f"""<div style="background-color: #d4edda; padding: 10px; border-radius: 5px; text-align: center;">
        <h3>Current Price</h3><p style="font-size: 24px; font-weight: bold;">${current_price:,.6f}</p></div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center;">
        <h3>EMA 20</h3><p style="font-size: 24px; font-weight: bold;">${ema_20:,.6f}</p></div>""", unsafe_allow_html=True)
    col3.markdown(f"""<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center;">
        <h3>EMA 50</h3><p style="font-size: 24px; font-weight: bold;">${ema_50:,.6f}</p></div>""", unsafe_allow_html=True)
    col4.markdown(f"""<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center;">
        <h3>EMA 100</h3><p style="font-size: 24px; font-weight: bold;">${ema_100:,.6f}</p></div>""", unsafe_allow_html=True)
    col5.markdown(f"""<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center;">
        <h3>EMA 200</h3><p style="font-size: 24px; font-weight: bold;">${ema_200:,.6f}</p></div>""", unsafe_allow_html=True)

    plot_data_with_ema(df)

except Exception as e:
    st.error(f"Error: {e}")

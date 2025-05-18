import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from collections import deque

st.set_page_config(layout="wide")

# -------------------- Globals --------------------
tick_buffer = deque(maxlen=500)
candle_df = pd.DataFrame()
ws = None

# -------------------- Plotting --------------------
def plot_candlestick_with_indicators(df, strategy):
    fig = go.Figure()

    if df.empty:
        return fig

    if strategy == "EMA Crossover":
        df["EMA_Fast"] = df["close"].ewm(span=5).mean()
        df["EMA_Slow"] = df["close"].ewm(span=10).mean()
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name='Candlestick'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], mode='lines', name='EMA Fast', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], mode='lines', name='EMA Slow', line=dict(color='orange')))

    elif strategy == "RSI":
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI"))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")

    else:
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name='Candlestick'))

    fig.update_layout(title=f"{strategy} Chart", xaxis_rangeslider_visible=False, height=600)
    return fig

# -------------------- Tick to Candle --------------------
def build_candles(ticks, interval="1m"):
    df = pd.DataFrame(ticks)
    df['epoch'] = pd.to_datetime(df['epoch'], unit='s')
    df.set_index('epoch', inplace=True)
    ohlc = df['quote'].resample(interval).ohlc().dropna()
    return ohlc

# -------------------- Async WebSocket --------------------
async def stream_ticks(app_id, symbol):
    global tick_buffer, candle_df

    uri = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    async with websockets.connect(uri) as websocket:
        st.success("✅ Connected to Deriv WebSocket")
        await websocket.send(json.dumps({"ticks_subscribe": symbol}))

        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if "tick" in data:
                tick = {
                    "epoch": data['tick']['epoch'],
                    "quote": float(data['tick']['quote'])
                }
                tick_buffer.append(tick)

                if len(tick_buffer) >= 10:  # buffer size to form a candle
                    candle_df = build_candles(tick_buffer)
                    st.session_state["df"] = candle_df
            await asyncio.sleep(0.5)

def run_async_ws(app_id, symbol):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream_ticks(app_id, symbol))

# -------------------- UI --------------------
st.title("📈 Deriv Live Strategy Dashboard")

col1, col2 = st.columns(2)
with col1:
    app_id = st.text_input("Enter your Deriv App ID", value="YOUR_APP_ID")
with col2:
    symbol = st.selectbox("Select Symbol", ["R_100", "R_50", "1HZ100V", "Volatility 75 (1s)"], index=0)

strategy = st.selectbox("Select Strategy", [
    "EMA Crossover", "RSI", "Basic Candlestick"
])

start_stream = st.button("Start Streaming")

if start_stream:
    if not app_id:
        st.warning("Please enter a valid App ID")
    else:
        st.info("⏳ Connecting and streaming...")
        st.session_state["df"] = pd.DataFrame()
        run_async_ws(app_id, symbol)

# -------------------- Live Chart --------------------
if "df" in st.session_state and not st.session_state["df"].empty:
    chart = plot_candlestick_with_indicators(st.session_state["df"], strategy)
    st.plotly_chart(chart, use_container_width=True)

import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from collections import deque

# -------------------- Config --------------------
st.set_page_config(layout="wide")
tick_buffer = deque(maxlen=500)
candle_df = pd.DataFrame()
ws = None
ws_task = None
subscribed = False

# -------------------- Strategy Charting --------------------
def plot_candlestick_with_indicators(df, strategy):
    fig = go.Figure()

    if df.empty:
        return fig

    if strategy == "EMA Crossover":
        df["EMA_Fast"] = df["close"].ewm(span=5).mean()
        df["EMA_Slow"] = df["close"].ewm(span=10).mean()
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name='Candlestick'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], name='EMA Fast', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], name='EMA Slow', line=dict(color='orange')))

    elif strategy == "RSI":
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI"))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")

    elif strategy == "MACD":
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        fig.add_trace(go.Scatter(x=df.index, y=macd, name="MACD", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=signal, name="Signal", line=dict(color='orange')))

    elif strategy == "Bollinger Bands":
        ma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        upper = ma + (2 * std)
        lower = ma - (2 * std)
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name='Candlestick'))
        fig.add_trace(go.Scatter(x=df.index, y=upper, name="Upper Band", line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df.index, y=lower, name="Lower Band", line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df.index, y=ma, name="MA", line=dict(color='gray')))

    else:
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name='Candlestick'))

    fig.update_layout(title=f"{strategy} Strategy", xaxis_rangeslider_visible=False, height=600)
    return fig

# -------------------- Data Handling --------------------
def build_candles(ticks, interval='1min'):
    df = pd.DataFrame(ticks)
    df['epoch'] = pd.to_datetime(df['epoch'], unit='s')
    df.set_index('epoch', inplace=True)
    ohlc = df['quote'].resample(interval).ohlc().dropna()
    return ohlc

async def fetch_history(app_id, symbol, count=100):
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "style": "ticks"
        }))
        response = await websocket.recv()
        data = json.loads(response)
        if "history" in data:
            return [{"epoch": int(e), "quote": float(q)} for e, q in zip(data['history']['times'], data['history']['prices'])]
    return []

async def stream_ticks(app_id, symbol, interval):
    global tick_buffer, candle_df, subscribed

    uri = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    async with websockets.connect(uri) as websocket:
        subscribed = True
        await websocket.send(json.dumps({"ticks_subscribe": symbol}))

        while subscribed:
            message = await websocket.recv()
            data = json.loads(message)

            if "tick" in data:
                tick = {
                    "epoch": data['tick']['epoch'],
                    "quote": float(data['tick']['quote'])
                }
                tick_buffer.append(tick)

                if len(tick_buffer) >= 10:
                    candle_df = build_candles(tick_buffer, interval)
                    st.session_state["df"] = candle_df
            await asyncio.sleep(0.5)

def start_stream(app_id, symbol, interval):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream_ticks(app_id, symbol, interval))

def stop_stream():
    global subscribed
    subscribed = False

# -------------------- UI --------------------
st.title("📊 Deriv Live Strategy Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    app_id = st.text_input("Deriv App ID", value="YOUR_APP_ID")
with col2:
    symbol = st.selectbox("Symbol", ["R_100", "R_50", "1HZ100V", "Volatility 75 (1s)"])
with col3:
    timeframe = st.selectbox("Timeframe", ["1min", "5min"], index=0)

strategy = st.selectbox("Select Strategy", [
    "EMA Crossover", "RSI", "MACD", "Bollinger Bands"
])

col_start, col_stop = st.columns(2)
if col_start.button("Start Stream"):
    if not app_id:
        st.warning("Please enter a valid App ID.")
    else:
        st.info("Fetching history...")
        history = asyncio.run(fetch_history(app_id, symbol))
        if history:
            tick_buffer.extend(history)
            st.session_state["df"] = build_candles(tick_buffer, timeframe)
        st.success("History loaded. Starting stream...")
        ws_task = asyncio.run(start_stream(app_id, symbol, timeframe))

if col_stop.button("Stop Stream"):
    stop_stream()
    st.success("Streaming stopped.")

# -------------------- Chart --------------------
if "df" in st.session_state and not st.session_state["df"].empty:
    chart = plot_candlestick_with_indicators(st.session_state["df"], strategy)
    st.plotly_chart(chart, use_container_width=True)

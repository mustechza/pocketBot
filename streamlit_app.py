import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import websocket, json, time
from threading import Thread
from datetime import datetime

# Streamlit setup
st.set_page_config(page_title="Binance WebSocket Signal App", layout="wide")
st.title("📡 Binance Live Signal Dashboard")

# Sidebar
symbol = st.sidebar.selectbox("Symbol", ["btcusdt", "ethusdt", "bnbusdt"], index=0)
interval = st.sidebar.selectbox("Interval", ["1m", "3m", "5m", "15m"], index=0)
refresh_rate = st.sidebar.slider("Refresh every (sec)", 5, 60, 15)

# Global state
if "klines" not in st.session_state:
    st.session_state.klines = []

# Start WebSocket
def start_kline_socket(symbol, interval, limit=100):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"

    def on_message(ws, message):
        data = json.loads(message)
        k = data['k']
        candle = {
            'time': pd.to_datetime(k['t'], unit='ms'),
            'open': float(k['o']),
            'high': float(k['h']),
            'low': float(k['l']),
            'close': float(k['c']),
            'volume': float(k['v'])
        }
        if st.session_state.klines and st.session_state.klines[-1]['time'] == candle['time']:
            st.session_state.klines[-1] = candle
        else:
            st.session_state.klines.append(candle)
            if len(st.session_state.klines) > limit:
                st.session_state.klines.pop(0)

    def on_error(ws, error): print("WebSocket error:", error)
    def on_close(ws): print("WebSocket closed")
    def on_open(ws): print("WebSocket connected")

    ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    thread = Thread(target=ws.run_forever)
    thread.daemon = True
    thread.start()

if "ws_started" not in st.session_state:
    start_kline_socket(symbol, interval)
    st.session_state.ws_started = True

# Wait for enough data
if len(st.session_state.klines) < 10:
    st.warning("Waiting for live candles...")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(st.session_state.klines)
df['ema_fast'] = df['close'].ewm(span=5, adjust=False).mean()
df['ema_slow'] = df['close'].ewm(span=13, adjust=False).mean()

# EMA crossover logic
def ema_signal(df):
    if df['ema_fast'].iloc[-2] < df['ema_slow'].iloc[-2] and df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1]:
        return "BUY"
    elif df['ema_fast'].iloc[-2] > df['ema_slow'].iloc[-2] and df['ema_fast'].iloc[-1] < df['ema_slow'].iloc[-1]:
        return "SELL"
    return None

signal = ema_signal(df)
timestamp = df['time'].iloc[-1]

# Signal log
def log_signal_to_file(signal):
    with open("signals.log", "a") as f:
        f.write(f"{datetime.now()} - {signal}\n")

if signal:
    st.success(f"**{signal} signal at {timestamp}**")
    log_signal_to_file(f"{signal} at {timestamp}")
else:
    st.info(f"No signal at {timestamp}")

# Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['time'], y=df['close'], name='Close', line=dict(color='white')))
fig.add_trace(go.Scatter(x=df['time'], y=df['ema_fast'], name='EMA 5', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df['time'], y=df['ema_slow'], name='EMA 13', line=dict(color='orange')))
fig.update_layout(title=f"{symbol.upper()} Live Price", height=400, xaxis_title="Time", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# Signal log viewer
with st.expander("📘 Signal Log"):
    try:
        with open("signals.log", "r") as f:
            st.text(f.read())
    except FileNotFoundError:
        st.info("No signals logged yet.")

# Refresh loop
time.sleep(refresh_rate)
st.experimental_rerun()

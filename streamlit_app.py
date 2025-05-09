import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from signal_strategies import ema_crossover_signal
from binance_ws import start_kline_socket, klines
from utils import log_signal, read_signal_log
import time

st.set_page_config(page_title="Binance Signal Dashboard", layout="wide")
st.title("📈 Binance Live EMA Signals")

symbol = st.sidebar.selectbox("Symbol", ["btcusdt", "ethusdt", "bnbusdt"])
interval = st.sidebar.selectbox("Interval", ["1m", "3m", "5m", "15m"])
refresh_rate = st.sidebar.slider("Refresh every (sec)", 5, 60, 15)

# Start WebSocket only once
if "started" not in st.session_state:
    start_kline_socket(symbol, interval)
    st.session_state.started = True

# Wait for candles
if len(klines) < 10:
    st.warning("Waiting for live data...")
    st.stop()

df = pd.DataFrame(klines)
signal = ema_crossover_signal(df)

if signal:
    st.success(f"{signal} signal at {df['time'].iloc[-1]}")
    log_signal(f"{signal} at {df['time'].iloc[-1]}")
else:
    st.info(f"No signal at {df['time'].iloc[-1]}")

# Plot
df['ema_fast'] = df['close'].ewm(span=5).mean()
df['ema_slow'] = df['close'].ewm(span=13).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['time'], y=df['close'], name="Close"))
fig.add_trace(go.Scatter(x=df['time'], y=df['ema_fast'], name="EMA 5"))
fig.add_trace(go.Scatter(x=df['time'], y=df['ema_slow'], name="EMA 13"))
fig.update_layout(height=400, title=f"{symbol.upper()} Live Chart", xaxis_title="Time")

st.plotly_chart(fig, use_container_width=True)

with st.expander("📘 Signal Log"):
    st.text(read_signal_log())

time.sleep(refresh_rate)
st.experimental_rerun()

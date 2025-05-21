import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import pytz
from datetime import datetime, timedelta
from utils import get_deriv_data, generate_signals, evaluate_signals, save_signal_to_csv

# App title and layout
st.set_page_config(page_title="Deriv Signal Bot", layout="wide")
st.title("📈 Deriv Live Signal Dashboard")

# Sidebar controls
symbol = st.sidebar.selectbox("Select Asset", ["R_50", "R_100", "R_75", "R_25"])
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m"])
strategy = st.sidebar.selectbox("Strategy", ["EMA_Cross", "RSI_OBOS", "MACD_Cross"])
trade_duration = st.sidebar.number_input("Trade Duration (minutes)", min_value=1, max_value=60, value=2)
period = st.sidebar.number_input("Backtest Period (candles)", min_value=50, max_value=500, value=200)

# Optional strategy parameters
strategy_params = {}
if strategy == "EMA_Cross":
    strategy_params["fast"] = st.sidebar.number_input("Fast EMA", 5, 50, 9)
    strategy_params["slow"] = st.sidebar.number_input("Slow EMA", 10, 200, 21)
elif strategy == "RSI_OBOS":
    strategy_params["rsi_period"] = st.sidebar.number_input("RSI Period", 5, 50, 14)
    strategy_params["overbought"] = st.sidebar.slider("Overbought Level", 60, 100, 70)
    strategy_params["oversold"] = st.sidebar.slider("Oversold Level", 0, 40, 30)
elif strategy == "MACD_Cross":
    strategy_params["fast"] = st.sidebar.number_input("MACD Fast", 5, 20, 12)
    strategy_params["slow"] = st.sidebar.number_input("MACD Slow", 10, 50, 26)
    strategy_params["signal"] = st.sidebar.number_input("MACD Signal", 5, 20, 9)

# Checkboxes
enable_backtest = st.sidebar.checkbox("Enable Backtest", value=True)
save_signals = st.sidebar.checkbox("Log Signals to CSV", value=True)

# Get live data
df = get_deriv_data(symbol, interval)
df.dropna(inplace=True)

# Generate signals
signals = generate_signals(df.copy(), strategy, **strategy_params)

# Show recent signals
st.subheader("📢 Latest Signals")
if not signals.empty:
    recent_signals = signals.tail(3)
    cols = st.columns(3)
    for i, row in recent_signals.iterrows():
        card = f"""
        ### {row['signal']} Signal
        - **Asset:** {symbol}
        - **Time:** {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        - **Price:** {row['price']:.2f}
        - **Expires in:** {trade_duration} min
        """
        cols[i % 3].markdown(card)

        if save_signals:
            save_signal_to_csv(row, symbol, strategy, trade_duration)
else:
    st.info("No signals found.")

# Plot chart with signals
st.subheader("📊 Price Chart")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index,
                             open=df['open'], high=df['high'],
                             low=df['low'], close=df['close'], name='Candles'))

buy_signals = signals[signals['signal'] == 'Buy']
sell_signals = signals[signals['signal'] == 'Sell']

fig.add_trace(go.Scatter(x=buy_signals['timestamp'], y=buy_signals['price'],
                         mode='markers', marker=dict(color='green', size=10),
                         name='Buy Signal'))
fig.add_trace(go.Scatter(x=sell_signals['timestamp'], y=sell_signals['price'],
                         mode='markers', marker=dict(color='red', size=10),
                         name='Sell Signal'))

fig.update_layout(height=500, margin=dict(t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

# Backtesting and stats
if enable_backtest:
    df_hist = get_deriv_data(symbol, interval, 200)
    signals_back = generate_signals(df_hist, strategy, **strategy_params)
    stats_back = evaluate_signals(df_hist, signals_back, period)

    st.subheader("📉 Backtest Performance")
    for key, val in stats_back.items():
        st.metric(label=key.replace('_', ' ').title(), value=round(val, 2) if isinstance(val, float) else val)

# Live performance stats
st.subheader("📈 Live Performance (Session)")
log_file = f"logs/{symbol}_{strategy}_live.csv"
try:
    df_log = pd.read_csv(log_file)
    df_log['timestamp'] = pd.to_datetime(df_log['timestamp'])
    df_log = df_log[df_log['timestamp'] >= datetime.now() - timedelta(hours=6)]
    stats_live = evaluate_signals(df, df_log, trade_duration)
    for key, val in stats_live.items():
        st.metric(label=key.replace('_', ' ').title(), value=round(val, 2) if isinstance(val, float) else val)
except FileNotFoundError:
    st.warning("No live performance data yet. Signals will be logged during session.")

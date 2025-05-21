import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import deque
import pytz
import os

# -------------------- Config --------------------
st.set_page_config(layout="wide")
st.title("📈 Deriv Strategy Signal Dashboard with Performance")

APP_ID = "76035"
TIMEZONE = pytz.timezone('Africa/Johannesburg')  # GMT+2

# -------------------- Session State Initialization --------------------
if "live_trades" not in st.session_state:
    st.session_state["live_trades"] = []

LOG_FILE = "live_trade_log.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        "timestamp", "symbol", "signal", "entry_time", "exit_time",
        "entry_price", "exit_price", "return_pct", "result"
    ]).to_csv(LOG_FILE, index=False)

# Placeholders
chart_placeholder = st.empty()
signals_placeholder = st.empty()
signal_log = deque(maxlen=3)

# -------------------- Sidebar Controls ----------------
mode = st.sidebar.radio("Mode", ["Live", "Backtest"], index=0)
asset = st.sidebar.selectbox("Select Asset", ["R_100", "R_50", "R_25", "R_10"])
strategy = st.sidebar.selectbox("Select Strategy", [
    "EMA Crossover", "RSI", "MACD", "Bollinger Bands", "Stochastic RSI", "Heikin-Ashi", "ATR Breakout"
])
show_confidence = st.sidebar.checkbox("Show Confidence %", True)
period = st.sidebar.number_input("Signal Duration (minutes)", min_value=1, max_value=60, value=2)
granularity = st.sidebar.selectbox("Granularity (seconds)", [60, 120, 300, 600], index=0)

# Strategy-specific parameters
ema_fast = st.sidebar.number_input("EMA Fast Span", min_value=2, max_value=100, value=5)
ema_slow = st.sidebar.number_input("EMA Slow Span", min_value=ema_fast+1, max_value=200, value=10)
rsi_period = st.sidebar.number_input("RSI Period", min_value=2, max_value=50, value=14)
macd_fast = st.sidebar.number_input("MACD Fast Span", min_value=2, max_value=50, value=12)
macd_slow = st.sidebar.number_input("MACD Slow Span", min_value=macd_fast+1, max_value=100, value=26)
macd_signal = st.sidebar.number_input("MACD Signal Span", min_value=1, max_value=30, value=9)
bb_window = st.sidebar.number_input("BB Window", min_value=2, max_value=100, value=20)
bb_std = st.sidebar.slider("BB Std Multiplier", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
stoch_period = st.sidebar.number_input("Stoch RSI Period", min_value=2, max_value=50, value=14)
atr_period = st.sidebar.number_input("ATR Period", min_value=2, max_value=50, value=14)

# Reset log button
if st.sidebar.button("Reset Performance Log"):
    os.remove(LOG_FILE)
    pd.DataFrame(columns=[
        "timestamp", "symbol", "signal", "entry_time", "exit_time",
        "entry_price", "exit_price", "return_pct", "result"
    ]).to_csv(LOG_FILE, index=False)
    st.sidebar.success("Performance log reset.")

# Backtest date inputs
if mode == "Backtest":
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    if st.sidebar.button("Run Backtest"):
        async def fetch_history():
            uri = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps({
                    "ticks_history": asset,
                    "adjust_start_time": 1,
                    "start": int(pd.Timestamp(start_date).timestamp()),
                    "end": int(pd.Timestamp(end_date).timestamp()),
                    "style": "candles",
                    "granularity": granularity,
                    "subscribe": 0
                }))
                msg = await ws.recv()
                data = json.loads(msg)
                return data.get('candles', [])

        hist = asyncio.run(fetch_history())
        df_hist = pd.DataFrame(hist)
        df_hist['epoch'] = pd.to_datetime(df_hist['epoch'], unit='s')
        df_hist = df_hist.set_index('epoch').tz_localize('UTC').tz_convert(TIMEZONE)
        df_hist.rename(columns={'open':'open','high':'high','low':'low','close':'close'}, inplace=True)

        fig_back, signals_back = plot_strategy(df_hist, strategy)
        chart_placeholder.plotly_chart(fig_back, use_container_width=True)

        # Show backtest signals
        st.subheader("🔁 Backtest Signals")
        for ts, sig, conf, price in signals_back:
            expiry = ts + pd.Timedelta(minutes=period)
            st.markdown(f"**{sig}** @ {price:.2f} — {ts.strftime('%Y-%m-%d %H:%M')} ➡️ {expiry.strftime('%Y-%m-%d %H:%M')} — 💡 {int(conf)}%")

        # Backtest performance metrics
def evaluate_signals(df, signals, duration_minutes):
    results = []
    for ts, signal, conf, entry_price in signals:
        exit_time = ts + pd.Timedelta(minutes=duration_minutes)
        if exit_time not in df.index:
            future = df[df.index > ts]
            if future.empty: continue
            exit_row = future.iloc[min(len(future)-1, duration_minutes)]
        else:
            exit_row = df.loc[exit_time]
        exit_price = exit_row['close']
        ret = ((exit_price - entry_price) / entry_price) * 100
        if signal == 'Sell': ret = -ret
        results.append(ret)
    arr = np.array(results)
    if arr.size == 0:
        return {"Total Trades":0,"Win Rate":0,"Avg Return %":0,"Max Drawdown %":0}
    win = (arr>0).sum()/arr.size*100
    avg = arr.mean()
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = (peak-cum).max()
    return {"Total Trades":len(arr),"Win Rate":round(win,2),"Avg Return %":round(avg,2),"Max Drawdown %":round(dd,2)}

        stats_back = evaluate_signals(df_hist, signals_back, period)
        st.subheader("📊 Backtest Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", stats_back["Total Trades"])
        c2.metric("Win Rate", f"{stats_back['Win Rate']}%")
        c3.metric("Avg Return", f"{stats_back['Avg Return %']}%")
        c4.metric("Max Drawdown", f"{stats_back['Max Drawdown %']}%")
        st.sidebar.success(f"Backtest done: {stats_back['Total Trades']} trades.")
        st.stop()

# Sidebar Metrics
st.sidebar.markdown("### ⚡️ Signal Metrics (Live)")

def plot_strategy(df, strategy):
    fig = go.Figure()
    signals = []
    if df.empty: return fig, signals
    # ... existing strategy logic here ...
    return fig, signals

# Live performance update & logging
def update_live_performance(df, symbol, duration):
    now = df.index[-1]
    completed = []
    open_trades = []
    for t in st.session_state['live_trades']:
        if now >= t['exit_time']:
            exit_idx = df.index.get_indexer([t['exit_time']], method='nearest')[0]
            exit_price = df.iloc[exit_idx]['close']
            ret = ((exit_price - t['entry_price'])/t['entry_price'])*100
            if t['signal']=='Sell': ret = -ret
            result = 'win' if ret>0 else 'loss'
            t.update({'exit_price':exit_price,'return_pct':round(ret,2),'result':result})
            # log CSV
            pd.DataFrame([{
                'timestamp': now, 'symbol': symbol,
                **t
            }]).to_csv(LOG_FILE, mode='a', index=False, header=False)
            completed.append(t)
        else:
            open_trades.append(t)
    st.session_state['live_trades'] = open_trades
    return completed

# Equity curve chart

def plot_equity_curve():
    df = pd.read_csv(LOG_FILE)
    if df.empty: return None
    df['cum_return'] = df['return_pct'].cumsum()
    fig = go.Figure(go.Scatter(x=pd.to_datetime(df['timestamp']), y=df['cum_return'], name='Equity'))
    fig.update_layout(title='Equity Curve', xaxis_title='Time', yaxis_title='Cumulative Return %', height=400)
    return fig

# Live streaming
async def stream_and_display():
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
    backoff = 1
    while True:
        try:
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps({
                    "ticks_history": asset,
                    "adjust_start_time": 1,
                    "count": 100,
                    "end": "latest",
                    "start": 1,
                    "style": "candles",
                    "granularity": granularity,
                    "subscribe": 1
                }))
                backoff = 1
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if 'candles' not in data: continue
                    df = pd.DataFrame(data['candles'])
                    df['epoch'] = pd.to_datetime(df['epoch'], unit='s')
                    df = df.set_index('epoch').tz_localize('UTC').tz_convert(TIMEZONE)
                    df.rename(columns={'open':'open','high':'high','low':'low','close':'close'}, inplace=True)

                    fig, signals = plot_strategy(df, strategy)
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

                    # Log new live signals
                    for ts, sig, conf, price in reversed(signals[-period:]):
                        entry_time = ts
                        exit_time = ts + pd.Timedelta(minutes=period)
                        st.session_state['live_trades'].append({
                            'signal': sig, 'entry_time': entry_time,
                            'exit_time': exit_time, 'entry_price': price
                        })
                        msg_md = f"**{sig}** @ {price:.2f}<br>{entry_time.strftime('%Y-%m-%d %H:%M:%S')} ➡️ {exit_time.strftime('%Y-%m-%d %H:%M:%S')}"
                        if msg_md not in signal_log:
                            signal_log.appendleft(msg_md)

                    # Display latest signals
                    with signals_placeholder.container():
                        st.subheader("📢 Latest Signals")
                        cols = st.columns(len(signal_log))
                        for i, m in enumerate(signal_log):
                            with cols[i]:
                                st.markdown(m, unsafe_allow_html=True)

                    # Sidebar counts
                    total = len(signal_log)
                    buys = sum('Buy' in m for m in signal_log)
                    st.sidebar.metric("Signals", total)
                    st.sidebar.metric("Buy", buys)
                    st.sidebar.metric("Sell", total-buys)

                    # Check and log completed trades
                    completed = update_live_performance(df, asset, period)
                    for tr in completed:
                        st.success(f"✔️ {tr['signal']} closed: {tr['return_pct']}% ({tr['result']})")

                    # Show live performance
                    st.subheader("📈 Live Performance Summary")
                    df_log = pd.read_csv(LOG_FILE)
                    if not df_log.empty:
                        win_rate = df_log['result'].eq('win').mean()*100
                        avg_ret = df_log['return_pct'].mean()
                        cum = df_log['return_pct'].cumsum()
                        max_dd = (cum.cummax()-cum).max()
                        c1,c2,c3,c4 = st.columns(4)
                        c1.metric("Total Trades", len(df_log))
                        c2.metric("Win Rate", f"{win_rate:.2f}%")
                        c3.metric("Avg Return", f"{avg_ret:.2f}%")
                        c4.metric("Max Drawdown", f"{max_dd:.2f}%")

                    # Equity curve
                    eq_fig = plot_equity_curve()
                    if eq_fig:
                        st.plotly_chart(eq_fig, use_container_width=True)

                    await asyncio.sleep(1)
        except Exception:
            await asyncio.sleep(backoff)
            backoff = min(backoff*2,60)

# -------------------- Run --------------------
if __name__ == '__main__':
    if mode == 'Live':
        asyncio.run(stream_and_display())
    else:
        pass  # backtest handled above

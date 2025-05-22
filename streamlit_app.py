import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import deque
import pytz
from datetime import datetime

# -------------------- Config --------------------
st.set_page_config(layout="wide")
st.title("📈 Deriv Signal Bot Dashboard")
APP_ID = "76035"
TIMEZONE = pytz.timezone('Africa/Johannesburg')  # GMT+2

# -------------------- UI Placeholders -----------------
chart_placeholder = st.empty()
signals_placeholder = st.empty()
signal_log = deque(maxlen=3)

# -------------------- Sidebar Controls ----------------
mode = st.sidebar.radio("Mode", ["Live", "Backtest"], index=0)
# Sidebar asset selection with tight-spread forex pairs
asset = st.sidebar.selectbox(
    "Select Asset",
    [
        # Major forex pairs (tight spreads)
        "frxEURUSD",  # Euro / US Dollar
        "frxGBPUSD",  # British Pound / US Dollar
        "frxUSDJPY",  # US Dollar / Japanese Yen
        "frxAUDUSD",  # Australian Dollar / US Dollar
        "frxUSDCHF",  # US Dollar / Swiss Franc
        "frxUSDCAD",  # US Dollar / Canadian Dollar
        "frxEURJPY",  # Euro / Japanese Yen
        "frxEURGBP",  # Euro / British Pound
        "frxGBPJPY",  # British Pound / Japanese Yen
        "frxAUDJPY",  # Australian Dollar / Japanese Yen

        # Optional synthetic indices (always open)
        "R_100", "R_50", "R_25", "R_10"
    ]
)strategies = st.sidebar.multiselect(
    "Select Strategies", 
    ["EMA Crossover", "RSI", "MACD", "Bollinger Bands", "Stochastic RSI", "Heikin-Ashi", "ATR Breakout"],
    default=["EMA Crossover", "RSI"]
)
show_confidence = st.sidebar.checkbox("Show Confidence %", True)
period = st.sidebar.number_input("Signal Duration (minutes)", min_value=1, max_value=60, value=2)
granularity = st.sidebar.selectbox("Granularity (seconds)", [60, 120, 300, 600], index=0)

# Strategy parameters (as before)
ema_fast = st.sidebar.number_input("EMA Fast Span", min_value=2, max_value=100, value=9)
ema_slow = st.sidebar.number_input("EMA Slow Span", min_value=ema_fast+1, max_value=200, value=21)
rsi_period = st.sidebar.number_input("RSI Period", min_value=2, max_value=50, value=14)
macd_fast = st.sidebar.number_input("MACD Fast Span", min_value=2, max_value=50, value=5)
macd_slow = st.sidebar.number_input("MACD Slow Span", min_value=macd_fast+1, max_value=100, value=13)
macd_signal = st.sidebar.number_input("MACD Signal Span", min_value=1, max_value=30, value=1)
bb_window = st.sidebar.number_input("BB Window", min_value=2, max_value=100, value=20)
bb_std = st.sidebar.slider("BB Std Multiplier", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
stoch_period = st.sidebar.number_input("Stoch RSI Period", min_value=2, max_value=50, value=14)
atr_period = st.sidebar.number_input("ATR Period", min_value=2, max_value=50, value=14)

# -------------------- Strategy Signal Extraction ----------------
def get_strategy_signals(df: pd.DataFrame, strat: str):
    df = df.copy()
    signals = []
    if df.empty:
        return signals
    if strat == "EMA Crossover":
        df['EMA_Fast'] = df['close'].ewm(span=ema_fast).mean()
        df['EMA_Slow'] = df['close'].ewm(span=ema_slow).mean()
        cond_buy = (df['EMA_Fast'] > df['EMA_Slow']) & (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1))
        cond_sell = (df['EMA_Fast'] < df['EMA_Slow']) & (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1))
        df['Signal'] = np.where(cond_buy, 'Buy', np.where(cond_sell, 'Sell', None))
        df['Confidence'] = (abs(df['EMA_Fast'] - df['EMA_Slow']) / df['close']) * 100
    elif strat == "RSI":
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(rsi_period).mean()
        loss = -delta.clip(upper=0).rolling(rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['Signal'] = np.where(df['RSI'] < 30, 'Buy', np.where(df['RSI'] > 70, 'Sell', None))
        df['Confidence'] = abs(df['RSI'] - 50) * 2
    elif strat == "MACD":
        exp1 = df['close'].ewm(span=macd_fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=macd_signal, adjust=False).mean()
        df['Signal'] = np.where(
            (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1)), 'Buy',
            np.where((macd < signal_line) & (macd.shift(1) >= signal_line.shift(1)), 'Sell', None)
        )
        avg_hist = (macd - signal_line).abs().rolling(20).mean()
        df['Confidence'] = ((macd - signal_line).abs() / avg_hist) * 50
        df['Confidence'] = df['Confidence'].clip(0, 100)
    elif strat == "Bollinger Bands":
        ma = df['close'].rolling(bb_window).mean()
        std = df['close'].rolling(bb_window).std()
        upper = ma + bb_std * std
        lower = ma - bb_std * std
        df['Signal'] = np.where(df['close'] < lower, 'Buy', np.where(df['close'] > upper, 'Sell', None))
        df['Confidence'] = (abs(df['close'] - ma) / (upper - lower)) * 100
    elif strat == "Stochastic RSI":
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(stoch_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(stoch_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        stoch = 100 * (rsi - rsi.rolling(stoch_period).min()) / (rsi.rolling(stoch_period).max() - rsi.rolling(stoch_period).min())
        df['Signal'] = np.where(stoch < 20, 'Buy', np.where(stoch > 80, 'Sell', None))
        df['Confidence'] = abs(stoch - 50) * 2
    elif strat == "Heikin-Ashi":
        ha = df.copy()
        ha['HA_Close'] = ha[['open', 'high', 'low', 'close']].mean(axis=1)
        ha['HA_Open'] = ha[['open', 'close']].shift(1).mean(axis=1)
        ha.loc[ha.index[0], 'HA_Open'] = ha['open'].iloc[0]
        ha['HA_High'] = ha[['HA_Open', 'HA_Close', 'high']].max(axis=1)
        ha['HA_Low'] = ha[['HA_Open', 'HA_Close', 'low']].min(axis=1)
        ha['Signal'] = np.where(ha['HA_Close'] > ha['HA_Open'], 'Buy', np.where(ha['HA_Close'] < ha['HA_Open'], 'Sell', None))
        ha['Confidence'] = abs(ha['HA_Close'] - ha['HA_Open']) / ha['HA_Close'] * 100
        df = ha
    elif strat == "ATR Breakout":
        tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
        atr = tr.rolling(atr_period).mean()
        upper = df['close'].shift(1) + atr
        lower = df['close'].shift(1) - atr
        df['Signal'] = np.where(df['close'] > upper, 'Buy', np.where(df['close'] < lower, 'Sell', None))
        df['Confidence'] = (abs(df['close'] - df['close'].shift(1)) / atr) * 50
    df['Confidence'] = df.get('Confidence', 0).fillna(0)
    for ts, row in df.iterrows():
        if pd.notna(row.get('Signal')):
            signals.append((ts, row['Signal'], min(max(row['Confidence'], 0), 100), row['close']))
    return signals

# -------------------- Multi-Strategy Aggregation & Plot ----------------
def plot_multi(df, selected_strats):
    # Base candlestick
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))

    # Collect per-strategy signals
    all_signals = []
    for strat in selected_strats:
        signals = get_strategy_signals(df, strat)
        all_signals.append({ 'strategy': strat, 'signals': signals })

    # Aggregate by timestamp
    agg = {}
    for s in all_signals:
        for ts, sig, conf, price in s['signals']:
            if ts not in agg:
                agg[ts] = {'buy': 0, 'sell': 0, 'confs': []}
            agg[ts][sig.lower()] += 1
            agg[ts]['confs'].append(conf)

    # Determine agreed signals
    agg_signals = []
    threshold = len(selected_strats) // 2 + 1
    for ts, v in agg.items():
        if v['buy'] >= threshold:
            agg_conf = sum(v['confs']) / len(v['confs'])
            price = df.loc[ts, 'close']
            agg_signals.append((ts, 'Buy', agg_conf, price))
        elif v['sell'] >= threshold:
            agg_conf = sum(v['confs']) / len(v['confs'])
            price = df.loc[ts, 'close']
            agg_signals.append((ts, 'Sell', agg_conf, price))

    # Plot aggregated markers
    for ts, sig, conf, price in agg_signals:
        fig.add_trace(go.Scatter(
            x=[ts], y=[price], mode='markers+text', text=sig,
            textposition='top center', marker=dict(size=12, symbol='triangle-up' if sig=='Buy' else 'triangle-down'),
            name='Agreed Signal'
        ))

    fig.update_layout(title=f"Multi-Strategy Agreed Signals on {asset}", height=600, xaxis_rangeslider_visible=False)
    return fig, agg_signals

# -------------------- Backtest Logic --------------------
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
                return json.loads(msg).get('candles', [])

        hist = asyncio.run(fetch_history())
        df_hist = pd.DataFrame(hist)
        df_hist['epoch'] = pd.to_datetime(df_hist['epoch'], unit='s')
        df_hist = df_hist.set_index('epoch').tz_localize('UTC').tz_convert(TIMEZONE)
        df_hist.rename(columns={'open':'open','high':'high','low':'low','close':'close'}, inplace=True)
        fig_back, signals_back = plot_multi(df_hist, strategies)
        chart_placeholder.plotly_chart(fig_back, use_container_width=True)
        st.subheader("🔁 Backtest Agreed Signals")
        for ts, sig, conf, price in signals_back:
            expiry = ts + pd.Timedelta(minutes=period)
            st.markdown(f"**{sig}** @ {price:.2f} — {ts.strftime('%Y-%m-%d %H:%M')} ➡️ {expiry.strftime('%Y-%m-%d %H:%M')} — 💡 {int(conf)}%")
        st.sidebar.success(f"Backtest found {len(signals_back)} agreed signals.")
        st.stop()

# -------------------- Live Mode --------------------
async def stream_live():
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
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
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if 'candles' not in data:
                continue
            df = pd.DataFrame(data['candles'])
            df['epoch'] = pd.to_datetime(df['epoch'], unit='s')
            df = df.set_index('epoch').tz_localize('UTC').tz_convert(TIMEZONE)
            df.rename(columns={'open':'open','high':'high','low':'low','close':'close'}, inplace=True)
            fig, signals = plot_multi(df, strategies)
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            with signals_placeholder.container():
                st.subheader("📡 Live Agreed Signals")
                for ts, sig, conf, price in reversed(signals[-3:]):
                    expiry = ts + pd.Timedelta(minutes=period)
                    st.markdown(f"**{sig}** @ {price:.2f} — {ts.strftime('%H:%M:%S')} ➡️ {expiry.strftime('%H:%M:%S')} — 💡 {int(conf)}%")

if mode == "Live":
    asyncio.run(stream_live())

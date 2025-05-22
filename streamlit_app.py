import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import deque
import pytz

# -------------------- Config --------------------
st.set_page_config(layout="wide")
st.title("📈 Deriv Strategy Signal Dashboard")

APP_ID = "76035"
TIMEZONE = pytz.timezone('Africa/Johannesburg')  # GMT+2

# -------------------- UI Placeholders --------------------
chart_placeholder = st.empty()
signals_placeholder = st.empty()
signal_log = deque(maxlen=3)

# -------------------- Sidebar Controls ----------------
mode = st.sidebar.radio("Mode", ["Live", "Backtest"], index=0)
asset = st.sidebar.selectbox("Select Asset", ["frxEURUSD", "frxGBPUSD", "frxUSDJPY", "R_100", "R_50", "R_25", "R_10"])
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

# -------------------- Strategy Logic ----------------
def plot_strategy(df, strategy):
    fig = go.Figure()
    signals = []

    if df.empty:
        return fig, signals

    if strategy == "EMA Crossover":
        df['EMA_Fast'] = df['close'].ewm(span=ema_fast).mean()
        df['EMA_Slow'] = df['close'].ewm(span=ema_slow).mean()
        df['Signal'] = np.where(
            (df['EMA_Fast'] > df['EMA_Slow']) & (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1)), 'Buy',
            np.where((df['EMA_Fast'] < df['EMA_Slow']) & (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1)), 'Sell', None)
        )
        df['Confidence'] = (abs(df['EMA_Fast'] - df['EMA_Slow']) / df['close']) * 100
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], name='EMA Fast'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], name='EMA Slow'))

    elif strategy == "RSI":
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(rsi_period).mean()
        loss = -delta.clip(upper=0).rolling(rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['Signal'] = np.where(df['RSI'] < 30, 'Buy', np.where(df['RSI'] > 70, 'Sell', None))
        df['Confidence'] = abs(df['RSI'] - 50) * 2
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        fig.add_hline(y=70, line_dash='dash')
        fig.add_hline(y=30, line_dash='dash')

    elif strategy == "MACD":
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
        fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=signal_line, name='Signal Line'))

    elif strategy == "Bollinger Bands":
        ma = df['close'].rolling(bb_window).mean()
        std = df['close'].rolling(bb_window).std()
        upper = ma + bb_std * std
        lower = ma - bb_std * std
        df['Signal'] = np.where(df['close'] < lower, 'Buy', np.where(df['close'] > upper, 'Sell', None))
        df['Confidence'] = (abs(df['close'] - ma) / (upper - lower)) * 100
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']))
        fig.add_trace(go.Scatter(x=df.index, y=upper, name='Upper Band'))
        fig.add_trace(go.Scatter(x=df.index, y=lower, name='Lower Band'))
        fig.add_trace(go.Scatter(x=df.index, y=ma, name='MA'))

    elif strategy == "Stochastic RSI":
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(stoch_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(stoch_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        stoch = 100 * (rsi - rsi.rolling(stoch_period).min()) / (rsi.rolling(stoch_period).max() - rsi.rolling(stoch_period).min())
        df['Signal'] = np.where(stoch < 20, 'Buy', np.where(stoch > 80, 'Sell', None))
        df['Confidence'] = abs(stoch - 50) * 2
        fig.add_trace(go.Scatter(x=df.index, y=stoch, name='Stoch RSI'))
        fig.add_hline(y=80, line_dash='dash')
        fig.add_hline(y=20, line_dash='dash')

    elif strategy == "Heikin-Ashi":
        ha = df.copy()
        ha['HA_Close'] = (df[['open', 'high', 'low', 'close']].sum(axis=1)) / 4
        ha['HA_Open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
        ha.loc[ha.index[0], 'HA_Open'] = df['open'].iloc[0]
        ha['HA_High'] = ha[['HA_Open', 'HA_Close', 'high']].max(axis=1)
        ha['HA_Low'] = ha[['HA_Open', 'HA_Close', 'low']].min(axis=1)
        df = ha
        df['Signal'] = np.where(df['HA_Close'] > df['HA_Open'], 'Buy', np.where(df['HA_Close'] < df['HA_Open'], 'Sell', None))
        df['Confidence'] = abs(df['HA_Close'] - df['HA_Open']) / df['HA_Close'] * 100
        fig.add_trace(go.Candlestick(x=df.index, open=df['HA_Open'], high=df['HA_High'], low=df['HA_Low'], close=df['HA_Close']))

    elif strategy == "ATR Breakout":
        tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
        atr = tr.rolling(atr_period).mean()
        upper = df['close'].shift(1) + atr
        lower = df['close'].shift(1) - atr
        df['Signal'] = np.where(df['close'] > upper, 'Buy', np.where(df['close'] < lower, 'Sell', None))
        df['Confidence'] = (abs(df['close'] - df['close'].shift(1)) / atr) * 50
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']))
        fig.add_trace(go.Scatter(x=df.index, y=upper, name='Upper Bound'))
        fig.add_trace(go.Scatter(x=df.index, y=lower, name='Lower Bound'))

    df['Confidence'] = df.get('Confidence', 0).fillna(0)
    df_signals = [(idx, row['Signal'], row['Confidence'], row['close']) for idx, row in df.iterrows() if pd.notna(row.get('Signal'))]

    for ts, sig, conf, price in df_signals:
        signals.append((ts, sig, min(max(conf, 0), 100), price))

    fig.update_layout(title=f"{strategy} on {asset}", height=600, xaxis_rangeslider_visible=False)
    return fig, signals

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
                data = json.loads(msg)
                return data.get('candles', [])

        hist = asyncio.run(fetch_history())
        df_hist = pd.DataFrame(hist)
        df_hist['epoch'] = pd.to_datetime(df_hist['epoch'], unit='s')
        df_hist = df_hist.set_index('epoch').tz_localize('UTC').tz_convert(TIMEZONE)
        df_hist.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
        fig_back, signals_back = plot_strategy(df_hist, strategy)
        chart_placeholder.plotly_chart(fig_back, use_container_width=True)

        st.subheader("🔁 Backtest Signals")
        for ts, sig, conf, price in signals_back:
            expiry = ts + pd.Timedelta(minutes=period)
            st.markdown(f"**{sig}** @ {price:.2f} — {ts.strftime('%Y-%m-%d %H:%M')} ➡️ {expiry.strftime('%Y-%m-%d %H:%M')} — 💡 {int(conf)}%")
        st.sidebar.success(f"Backtest found {len(signals_back)} signals.")
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
            df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
            fig, signals = plot_strategy(df, strategy)
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            with signals_placeholder.container():
                st.subheader("📡 Live Signals")
                for ts, sig, conf, price in reversed(signals[-3:]):
                    expiry = ts + pd.Timedelta(minutes=period)
                    st.markdown(f"**{sig}** @ {price:.2f} — {ts.strftime('%H:%M:%S')} ➡️ {expiry.strftime('%H:%M:%S')} — 💡 {int(conf)}%")

if mode == "Live":
    asyncio.run(stream_live())

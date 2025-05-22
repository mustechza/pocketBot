import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import deque, defaultdict
import pytz
from datetime import datetime

# -------------------- Config --------------------
st.set_page_config(layout="wide")
st.title("📈 Deriv Multi-Strategy Signal Bot Dashboard")

APP_ID = "76035"
TIMEZONE = pytz.timezone('Africa/Johannesburg')

chart_placeholder = st.empty()
signals_placeholder = st.empty()

# -------------------- Sidebar --------------------
mode = st.sidebar.radio("Mode", ["Live", "Backtest"], index=0)
asset = st.sidebar.selectbox("Select Asset", ["frxEURUSD", "frxGBPUSD", "frxUSDJPY", "R_100", "R_50", "R_25", "R_10"])
granularity = st.sidebar.selectbox("Granularity (seconds)", [60, 120, 300, 600], index=0)
period = st.sidebar.number_input("Signal Duration (minutes)", min_value=1, max_value=60, value=2)
show_confidence = st.sidebar.checkbox("Show Confidence %", True)

# Multi-strategy enable toggles
strategies_enabled = {
    "EMA Crossover": st.sidebar.checkbox("EMA Crossover", True),
    "RSI": st.sidebar.checkbox("RSI", True),
    "MACD": st.sidebar.checkbox("MACD", True),
    "Bollinger Bands": st.sidebar.checkbox("Bollinger Bands", True),
    "Stochastic RSI": st.sidebar.checkbox("Stochastic RSI", False),
    "Heikin-Ashi": st.sidebar.checkbox("Heikin-Ashi", False),
    "ATR Breakout": st.sidebar.checkbox("ATR Breakout", False),
}

# Strategy params (can also be sidebar inputs, omitted here for brevity)
ema_fast, ema_slow = 5, 10
rsi_period = 14
macd_fast, macd_slow, macd_signal = 12, 26, 9
bb_window, bb_std = 20, 2.0
stoch_period = 14
atr_period = 14

# -------------------- Strategy Functions --------------------
def ema_crossover(df):
    df['EMA_Fast'] = df['close'].ewm(span=ema_fast).mean()
    df['EMA_Slow'] = df['close'].ewm(span=ema_slow).mean()
    df['Signal'] = np.where(
        (df['EMA_Fast'] > df['EMA_Slow']) & (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1)), 'Buy',
        np.where((df['EMA_Fast'] < df['EMA_Slow']) & (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1)), 'Sell', None)
    )
    df['Confidence'] = (abs(df['EMA_Fast'] - df['EMA_Slow']) / df['close']) * 100
    return df[['Signal', 'Confidence']]

def rsi_strategy(df):
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Signal'] = np.where(df['RSI'] < 30, 'Buy', np.where(df['RSI'] > 70, 'Sell', None))
    df['Confidence'] = abs(df['RSI'] - 50) * 2
    return df[['Signal', 'Confidence']]

def macd_strategy(df):
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
    return df[['Signal', 'Confidence']]

def bollinger_strategy(df):
    ma = df['close'].rolling(bb_window).mean()
    std = df['close'].rolling(bb_window).std()
    upper = ma + bb_std * std
    lower = ma - bb_std * std
    df['Signal'] = np.where(df['close'] < lower, 'Buy', np.where(df['close'] > upper, 'Sell', None))
    df['Confidence'] = (abs(df['close'] - ma) / (upper - lower)) * 100
    return df[['Signal', 'Confidence']]

# Additional strategies omitted for brevity, implement similarly...

# -------------------- Multi-Strategy Signal Aggregation --------------------
def run_strategies(df):
    signals = {}
    if strategies_enabled["EMA Crossover"]:
        signals["EMA"] = ema_crossover(df)
    if strategies_enabled["RSI"]:
        signals["RSI"] = rsi_strategy(df)
    if strategies_enabled["MACD"]:
        signals["MACD"] = macd_strategy(df)
    if strategies_enabled["Bollinger Bands"]:
        signals["BB"] = bollinger_strategy(df)
    # Add calls to other strategies if enabled here...

    # Combine signals per timestamp
    combined = pd.DataFrame(index=df.index)
    combined['Buy'] = 0
    combined['Sell'] = 0
    combined['None'] = 0
    combined['ConfidenceSum'] = 0
    combined['SignalCount'] = 0

    for strat_name, strat_df in signals.items():
        for idx, row in strat_df.iterrows():
            sig = row['Signal']
            conf = row['Confidence'] if not np.isnan(row['Confidence']) else 0
            if sig == 'Buy':
                combined.at[idx, 'Buy'] += 1
                combined.at[idx, 'ConfidenceSum'] += conf
                combined.at[idx, 'SignalCount'] += 1
            elif sig == 'Sell':
                combined.at[idx, 'Sell'] += 1
                combined.at[idx, 'ConfidenceSum'] += conf
                combined.at[idx, 'SignalCount'] += 1
            else:
                combined.at[idx, 'None'] += 1

    # Majority vote for each timestamp
    def majority_signal(row):
        if row['Buy'] > row['Sell'] and row['Buy'] >= row['SignalCount'] / 2:
            return "Buy"
        elif row['Sell'] > row['Buy'] and row['Sell'] >= row['SignalCount'] / 2:
            return "Sell"
        else:
            return None

    combined['ConsensusSignal'] = combined.apply(majority_signal, axis=1)

    # Average confidence of agreeing signals
    combined['AvgConfidence'] = combined.apply(
        lambda r: (r['ConfidenceSum'] / (r['Buy'] + r['Sell'])) if (r['Buy'] + r['Sell']) > 0 else 0,
        axis=1)

    # Filter only timestamps with consensus signals
    consensus_signals = combined[combined['ConsensusSignal'].notnull()]

    return consensus_signals

# -------------------- Plot & Display --------------------
def plot_candles(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'])])
    fig.update_layout(title=f"{asset} Candlestick Chart", xaxis_rangeslider_visible=False, height=600)
    return fig

# -------------------- Main --------------------
if mode == "Backtest":
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    if st.sidebar.button("Run Backtest"):
        # Fetch historical candles (similar to previous example)
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
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if 'candles' in data:
                        return data['candles']

        hist = asyncio.run(fetch_history())
        df = pd.DataFrame(hist)
        df['epoch'] = pd.to_datetime(df['epoch'], unit='s').dt.tz_localize('UTC').dt.tz_convert(TIMEZONE)
        df.set_index('epoch', inplace=True)
        df = df.astype(float)

        consensus = run_strategies(df)
        fig = plot_candles(df)
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        signals_placeholder.markdown("### Consensus Signals (Backtest)")
        for ts, row in consensus.tail(10).iterrows():
            conf_text = f" - Confidence: {row['AvgConfidence']:.1f}%" if show_confidence else ""
            signals_placeholder.write(f"{ts.strftime('%Y-%m-%d %H:%M:%S')}: **{row['ConsensusSignal']}**{conf_text}")

else:
    # Live mode similar to your existing code but calling run_strategies on live data
    import threading

    loop = asyncio.new_event_loop()
    ws_task = None
    stop_stream = False

    live_candles = deque(maxlen=200)

    start_button = st.sidebar.button("Start Live Stream")
    stop_button = st.sidebar.button("Stop Live Stream")

    def start_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    if start_button and not ws_task:
        stop_stream = False
        threading.Thread(target=start_loop, args=(loop,), daemon=True).start()

        async def live_stream():
            nonlocal stop_stream
            uri = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
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
                    while not stop_stream:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        if "candles" in data:
                            for candle in data["candles"]:
                                candle['epoch'] = pd.to_datetime(candle['epoch'], unit='s').tz_localize('UTC').tz_convert(TIMEZONE)
                                candle['open'] = float(candle['open'])
                                candle['high'] = float(candle['high'])
                                candle['low'] = float(candle['low'])
                                candle['close'] = float(candle['close'])
                                candle['volume'] = float(candle['volume'])
                                live_candles.append(candle)
                        await asyncio.sleep(0.1)
            except Exception as e:
                st.error(f"Live stream error: {e}")

        ws_task = loop.create_task(live_stream())

    if stop_button and ws_task:
        stop_stream = True
        ws_task.cancel()
        ws_task = None
        signals_placeholder.markdown("### Stream stopped")

    if live_candles:
        df_live = pd.DataFrame(live_candles).set_index('epoch')
        consensus = run_strategies(df_live)
        fig = plot_candles(df_live)
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        signals_placeholder.markdown("### Latest Consensus Signals")
        for ts, row in consensus.tail(5).iterrows():
            conf_text = f" - Confidence: {row['AvgConfidence']:.1f}%" if show_confidence else ""
            signals_placeholder.write(f"{ts.strftime('%Y-%m-%d %H:%M:%S')}: **{row['ConsensusSignal']}**{conf_text}")

    if ws_task:
        st.experimental_rerun()

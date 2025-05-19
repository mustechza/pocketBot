import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from collections import deque
import pytz
import time

# -------------------- Config --------------------
st.set_page_config(layout="wide")
st.title("Deriv Strategy Signal Dashboard")

APP_ID = "76035"
TIMEZONE = pytz.timezone('Africa/Johannesburg')  # GMT+2

# -------------------- Session State --------------------
if "candle_df" not in st.session_state:
    st.session_state["candle_df"] = pd.DataFrame()
if "signal_log" not in st.session_state:
    st.session_state["signal_log"] = deque(maxlen=3)

# -------------------- Sidebar Controls --------------------
asset = st.sidebar.selectbox("Select Asset", ["R_100", "R_50", "R_25", "R_10"])
strategy = st.sidebar.selectbox("Select Strategy", [
    "EMA Crossover", "RSI", "MACD", "Bollinger Bands", "Stochastic RSI", "Heikin-Ashi Reversal"
])
granularity_map = {"1m": 60, "5m": 300}
granularity_label = st.sidebar.selectbox("Granularity", list(granularity_map.keys()))
granularity = granularity_map[granularity_label]
show_confidence = st.sidebar.checkbox("Show Confidence %", True)

if st.sidebar.button("\ud83d\uded1 Stop Stream"):
    st.stop()

# -------------------- Strategy Logic --------------------
def apply_strategy(df, strategy):
    fig = go.Figure()
    signals = []

    if df.empty or len(df) < 20 or df.isnull().values.any():
        return fig, signals

    if strategy == "EMA Crossover":
        df['EMA_Fast'] = df['close'].ewm(span=5).mean()
        df['EMA_Slow'] = df['close'].ewm(span=10).mean()
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
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['Signal'] = np.where(df['RSI'] < 30, 'Buy', np.where(df['RSI'] > 70, 'Sell', None))
        df['Confidence'] = abs(df['RSI'] - 50) * 2
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        fig.add_hline(y=70, line_dash='dash')
        fig.add_hline(y=30, line_dash='dash')

    elif strategy == "MACD":
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()
        df['MACD'] = macd
        df['SignalLine'] = signal_line
        df['Histogram'] = macd - signal_line
        df['Signal'] = np.where(
            (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1)), 'Buy',
            np.where((macd < signal_line) & (macd.shift(1) >= signal_line.shift(1)), 'Sell', None)
        )
        avg_hist = df['Histogram'].abs().rolling(20).mean()
        df['Confidence'] = (df['Histogram'].abs() / avg_hist) * 50
        df['Confidence'] = df['Confidence'].clip(0, 100)
        fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=signal_line, name='Signal Line'))

    elif strategy == "Bollinger Bands":
        ma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        df['Upper'], df['Lower'], df['MA'] = upper, lower, ma
        df['Signal'] = np.where(df['close'] < lower, 'Buy', np.where(df['close'] > upper, 'Sell', None))
        band_width = upper - lower
        df['Confidence'] = (abs(df['close'] - ma) / band_width) * 100
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']))
        fig.add_trace(go.Scatter(x=df.index, y=upper, name='Upper Band'))
        fig.add_trace(go.Scatter(x=df.index, y=lower, name='Lower Band'))
        fig.add_trace(go.Scatter(x=df.index, y=ma, name='MA'))

    elif strategy == "Stochastic RSI":
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        stoch_rsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
        df['StochRSI'] = stoch_rsi
        df['Signal'] = np.where(stoch_rsi < 0.2, 'Buy', np.where(stoch_rsi > 0.8, 'Sell', None))
        df['Confidence'] = abs(stoch_rsi - 0.5) * 200
        fig.add_trace(go.Scatter(x=df.index, y=stoch_rsi, name='Stochastic RSI'))

    elif strategy == "Heikin-Ashi Reversal":
        ha_df = df.copy()
        ha_df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_df['HA_Open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
        ha_df['HA_Open'].fillna(df['open'], inplace=True)
        df['Signal'] = np.where((ha_df['HA_Close'] > ha_df['HA_Open']) & (ha_df['HA_Close'].shift(1) < ha_df['HA_Open'].shift(1)), 'Buy',
                                np.where((ha_df['HA_Close'] < ha_df['HA_Open']) & (ha_df['HA_Close'].shift(1) > ha_df['HA_Open'].shift(1)), 'Sell', None))
        df['Confidence'] = abs(ha_df['HA_Close'] - ha_df['HA_Open']) / df['close'] * 100
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']))

    df['Confidence'] = df['Confidence'].fillna(0)

    for ts, row in df[df['Signal'].notna()].iterrows():
        conf = min(max(row['Confidence'], 0), 100)
        signals.append((ts, row['Signal'], conf, row['close']))
        fig.add_trace(go.Scatter(
            x=[ts], y=[row['close']],
            mode='markers+text',
            marker=dict(color='green' if row['Signal']=='Buy' else 'red', size=10,
                        symbol='arrow-up' if row['Signal']=='Buy' else 'arrow-down'),
            text=[f"{row['Signal']} {int(conf)}%" if show_confidence else row['Signal']],
            textposition='top center' if row['Signal']=='Buy' else 'bottom center',
            showlegend=False
        ))

    fig.update_layout(title=f"{strategy} Strategy on {asset} ({granularity_label})", height=600, xaxis_rangeslider_visible=False)
    return fig, signals

# -------------------- WebSocket Listener --------------------
async def deriv_ws_listener():
    retries = 5
    for attempt in range(retries):
        try:
            async with websockets.connect("wss://ws.derivws.com/websockets/v3?app_id=" + APP_ID) as ws:
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
                    if 'candles' in data:
                        df = pd.DataFrame(data['candles'])
                        df['epoch'] = pd.to_datetime(df['epoch'], unit='s').dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
                        df.set_index('epoch', inplace=True)
                        df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
                        st.session_state["candle_df"] = df
                        return
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            await asyncio.sleep(1)

# -------------------- Main --------------------
async def main():
    await deriv_ws_listener()
    fig, signals = apply_strategy(st.session_state["candle_df"].copy(), strategy)
    st.plotly_chart(fig, use_container_width=True)

    for ts, sig, conf, price in reversed(signals[-3:]):
        ts_fmt = ts.strftime('%Y-%m-%d %H:%M:%S')
        st.session_state["signal_log"].appendleft(f"**{sig}** @ {price:.2f} — {ts_fmt} — \ud83d\udca1 Confidence: `{int(conf)}%`")

    st.subheader("\ud83d\udce2 Latest Signals")
    cols = st.columns(3)
    for i, msg in enumerate(st.session_state["signal_log"]):
        with cols[i]:
            st.markdown(msg, unsafe_allow_html=True)

# -------------------- Run --------------------
if __name__ == "__main__":
    asyncio.run(main())

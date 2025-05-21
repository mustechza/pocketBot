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

# -------------------- Config --------------------
st.set_page_config(layout="wide")
st.title("📈 Deriv Strategy Signal Dashboard")

APP_ID = "76035"
TIMEZONE = pytz.timezone('Africa/Johannesburg')  # GMT+2
tick_buffer = deque(maxlen=100)
candle_df = pd.DataFrame()
signal_log = deque(maxlen=3)

# -------------------- Sidebar Controls --------------------
asset = st.sidebar.selectbox(
    "Select Asset", 
    ["R_100", "R_50", "R_25", "R_10", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"]
)
strategy = st.sidebar.selectbox("Select Strategy", [
    "EMA Crossover", "RSI", "MACD", "Bollinger Bands", "Stochastic RSI", "Heikin-Ashi", "ATR Breakout"])
show_confidence = st.sidebar.checkbox("Show Confidence %", True)

# -------------------- Strategy Logic --------------------
def plot_strategy(df, strategy):
    fig = go.Figure()
    signals = []

    if df.empty:
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
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        stoch_rsi = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)
        df['StochRSI'] = stoch_rsi
        df['Signal'] = np.where(stoch_rsi < 20, 'Buy', np.where(stoch_rsi > 80, 'Sell', None))
        df['Confidence'] = abs(stoch_rsi - 50) * 2

        fig.add_trace(go.Scatter(x=df.index, y=df['StochRSI'], name='Stoch RSI'))
        fig.add_hline(y=80, line_dash='dash')
        fig.add_hline(y=20, line_dash='dash')

    elif strategy == "Heikin-Ashi":
        ha_df = df.copy()
        ha_df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_df['HA_Open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
        ha_df['HA_Open'].iloc[0] = df['open'].iloc[0]  # seed first value
        for i in range(1, len(ha_df)):
            ha_df['HA_Open'].iloc[i] = (ha_df['HA_Open'].iloc[i - 1] + ha_df['HA_Close'].iloc[i - 1]) / 2
        ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close', 'high']].max(axis=1)
        ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close', 'low']].min(axis=1)
        df = ha_df
        df['Signal'] = np.where(df['HA_Close'] > df['HA_Open'], 'Buy', np.where(df['HA_Close'] < df['HA_Open'], 'Sell', None))
        df['Confidence'] = abs(df['HA_Close'] - df['HA_Open']) / df['HA_Close'] * 100

        fig.add_trace(go.Candlestick(x=df.index, open=df['HA_Open'], high=df['HA_High'], low=df['HA_Low'], close=df['HA_Close']))

    elif strategy == "ATR Breakout":
        df['TR'] = np.maximum((df['high'] - df['low']), np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
        df['ATR'] = df['TR'].rolling(14).mean()
        df['Upper'] = df['close'].shift(1) + df['ATR']
        df['Lower'] = df['close'].shift(1) - df['ATR']
        df['Signal'] = np.where(df['close'] > df['Upper'], 'Buy', np.where(df['close'] < df['Lower'], 'Sell', None))
        df['Confidence'] = (abs(df['close'] - df['close'].shift(1)) / df['ATR']) * 50

        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='Upper Bound'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='Lower Bound'))

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

    fig.update_layout(title=f"{strategy} Strategy on {asset}", height=600, xaxis_rangeslider_visible=False)
    return fig, signals

# -------------------- WebSocket Listener --------------------
async def deriv_ws_listener():
    global candle_df
    async with websockets.connect("wss://ws.derivws.com/websockets/v3?app_id=" + APP_ID) as ws:
        await ws.send(json.dumps({
            "ticks_history": asset,
            "adjust_start_time": 1,
            "count": 100,
            "end": "latest",
            "start": 1,
            "style": "candles",
            "granularity": 60,
            "subscribe": 1
        }))
        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                if 'candles' in data:
                    candles = data['candles']
                    df = pd.DataFrame(candles)
                    df['epoch'] = pd.to_datetime(df['epoch'], unit='s').dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
                    df.set_index('epoch', inplace=True)
                    df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
                    candle_df = df
                    return
            except:
                continue

# -------------------- Main Layout --------------------
async def main():
    await deriv_ws_listener()
    fig, signals = plot_strategy(candle_df.copy(), strategy)
    st.plotly_chart(fig, use_container_width=True)

    for ts, sig, conf, price in reversed(signals[-3:]):
        ts_fmt = ts.strftime('%Y-%m-%d %H:%M:%S')
        signal_log.appendleft(f"**{sig}** @ {price:.2f} — {ts_fmt} — 💡 Confidence: `{int(conf)}%`")

    with st.container():
        st.subheader("📢 Latest Signals")
        cols = st.columns(3)
        for i, msg in enumerate(signal_log):
            with cols[i]:
                st.markdown(msg, unsafe_allow_html=True)

# -------------------- Run --------------------
if __name__ == "__main__":
    asyncio.run(main())

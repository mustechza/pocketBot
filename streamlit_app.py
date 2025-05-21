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

# Placeholders
chart_placeholder = st.empty()
signals_placeholder = st.empty()
signal_log = deque(maxlen=3)

# Sidebar Metrics Placeholders (auto-refresh)
st.sidebar.markdown("### ⚡️ Signal Metrics")

# -------------------- Sidebar Controls ----------------
asset = st.sidebar.selectbox("Select Asset", ["R_100", "R_50", "R_25", "R_10"])
strategy = st.sidebar.selectbox("Select Strategy", [
    "EMA Crossover", "RSI", "MACD", "Bollinger Bands", "Stochastic RSI", "Heikin-Ashi", "ATR Breakout"
])
show_confidence = st.sidebar.checkbox("Show Confidence %", True)
period = st.sidebar.number_input("Signal Duration (minutes)", min_value=1, max_value=60, value=2)
granularity = st.sidebar.selectbox("Granularity (seconds)", [60, 120, 300, 600], index=0)

# -------------------- Strategy Logic --------------------
def plot_strategy(df, strategy):
    fig = go.Figure()
    signals = []

    if df.empty:
        return fig, signals

    # EMA Crossover
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

    # RSI
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

    # MACD
    elif strategy == "MACD":
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()
        df['Signal'] = np.where(
            (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1)), 'Buy',
            np.where((macd < signal_line) & (macd.shift(1) >= signal_line.shift(1)), 'Sell', None)
        )
        avg_hist = (macd - signal_line).abs().rolling(20).mean()
        df['Confidence'] = ((macd - signal_line).abs() / avg_hist) * 50
        df['Confidence'] = df['Confidence'].clip(0, 100)
        fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=signal_line, name='Signal Line'))

    # Bollinger Bands
    elif strategy == "Bollinger Bands":
        ma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        df['Signal'] = np.where(df['close'] < lower, 'Buy', np.where(df['close'] > upper, 'Sell', None))
        df['Confidence'] = (abs(df['close'] - ma) / (upper - lower)) * 100
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']))
        fig.add_trace(go.Scatter(x=df.index, y=upper, name='Upper Band'))
        fig.add_trace(go.Scatter(x=df.index, y=lower, name='Lower Band'))
        fig.add_trace(go.Scatter(x=df.index, y=ma, name='MA'))

    # Stochastic RSI
    elif strategy == "Stochastic RSI":
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        stoch = 100 * (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
        df['Signal'] = np.where(stoch < 20, 'Buy', np.where(stoch > 80, 'Sell', None))
        df['Confidence'] = abs(stoch - 50) * 2
        fig.add_trace(go.Scatter(x=df.index, y=stoch, name='Stoch RSI'))
        fig.add_hline(y=80, line_dash='dash')
        fig.add_hline(y=20, line_dash='dash')

    # Heikin-Ashi
    elif strategy == "Heikin-Ashi":
        ha = df.copy()
        ha['HA_Close'] = (df[['open','high','low','close']].sum(axis=1)) / 4
        ha['HA_Open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
        ha.loc[ha.index[0], 'HA_Open'] = df['open'].iloc[0]
        ha['HA_High'] = ha[['HA_Open','HA_Close','high']].max(axis=1)
        ha['HA_Low'] = ha[['HA_Open','HA_Close','low']].min(axis=1)
        df = ha
        df['Signal'] = np.where(df['HA_Close'] > df['HA_Open'], 'Buy', np.where(df['HA_Close'] < df['HA_Open'], 'Sell', None))
        df['Confidence'] = abs(df['HA_Close'] - df['HA_Open']) / df['HA_Close'] * 100
        fig.add_trace(go.Candlestick(x=df.index, open=df['HA_Open'], high=df['HA_High'], low=df['HA_Low'], close=df['HA_Close']))

    # ATR Breakout
    elif strategy == "ATR Breakout":
        tr = np.maximum((df['high'] - df['low']),
                        np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
        atr = tr.rolling(14).mean()
        upper = df['close'].shift(1) + atr
        lower = df['close'].shift(1) - atr
        df['Signal'] = np.where(df['close'] > upper, 'Buy', np.where(df['close'] < lower, 'Sell', None))
        df['Confidence'] = (abs(df['close'] - df['close'].shift(1)) / atr) * 50
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']))
        fig.add_trace(go.Scatter(x=df.index, y=upper, name='Upper Bound'))
        fig.add_trace(go.Scatter(x=df.index, y=lower, name='Lower Bound'))

    # Finalize
    df['Confidence'] = df.get('Confidence', 0).fillna(0)
    df_signals = [(idx, row['Signal'], row['Confidence'], row['close'])
                  for idx, row in df.iterrows() if pd.notna(row.get('Signal'))]

    for ts, sig, conf, price in df_signals:
        if ts and price:
            signals.append((ts, sig, min(max(conf, 0), 100), price))

    fig.update_layout(title=f"{strategy} on {asset}", height=600, xaxis_rangeslider_visible=False)
    return fig, signals

# -------------------- Async Stream Loop with Reconnect --------------------
async def stream_and_display():
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
    backoff = 1
    while True:
        try:
            async with websockets.connect(uri) as ws:
                # subscribe
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
                backoff = 1  # reset on success

                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if 'candles' not in data:
                        continue
                    df = pd.DataFrame(data['candles'])
                    df['epoch'] = pd.to_datetime(df['epoch'], unit='s')
                    df = df.set_index('epoch').tz_localize('UTC').tz_convert(TIMEZONE)
                    df.rename(columns={'open':'open','high':'high','low':'low','close':'close'}, inplace=True)

                    fig, signals = plot_strategy(df, strategy)
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

                    # update signal log
                    for ts, sig, conf, price in reversed(signals[-period:]):
                        ts_fmt = ts.strftime('%Y-%m-%d %H:%M:%S')
                        expiry_fmt = (ts + pd.Timedelta(minutes=period)).strftime('%Y-%m-%d %H:%M:%S')
                        confidence_part = f" — 💡 Confidence: `{int(conf)}%`" if show_confidence else ""
                        msg_md = f"**{sig}** @ {price:.2f}<br>{ts_fmt} ➡️ {expiry_fmt}{confidence_part}"
                        if msg_md not in signal_log:
                            signal_log.appendleft(msg_md)

                    # render signals
                    with signals_placeholder.container():
                        st.subheader("📢 Latest Signals")
                        cols = st.columns(len(signal_log))
                        for i, m in enumerate(signal_log):
                            with cols[i]:
                                st.markdown(m, unsafe_allow_html=True)

                    # update sidebar metrics
                    total_signals = len(signal_log)
                    buy_count = sum(1 for m in signal_log if 'Buy' in m)
                    sell_count = total_signals - buy_count
                    st.sidebar.metric("Signals", total_signals)
                    st.sidebar.metric("Buy", buy_count)
                    st.sidebar.metric("Sell", sell_count)

                    await asyncio.sleep(1)

        except Exception:
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue

# -------------------- Run --------------------
if __name__ == '__main__':
    asyncio.run(stream_and_display())

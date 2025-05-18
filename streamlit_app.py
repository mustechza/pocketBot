import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from collections import deque
import pytz

# -------------------- Config --------------------
st.set_page_config(layout="wide")
tick_buffer = deque(maxlen=500)
candle_df = pd.DataFrame()
subscribed = False
APP_ID = "76035"  # Hardcoded App ID
TIMEZONE = pytz.timezone('Africa/Johannesburg')  # GMT+2

# -------------------- Strategy Charting with Signals & Confidence --------------------
def plot_candlestick_with_indicators(df, strategy):
    fig = go.Figure()
    signals = []
    if df.empty:
        return fig, signals

    # compute indicators & scores
    if strategy == "EMA Crossover":
        df['EMA_Fast'] = df['close'].ewm(span=5).mean()
        df['EMA_Slow'] = df['close'].ewm(span=10).mean()
        df['Signal'] = np.where(
            (df['EMA_Fast'] > df['EMA_Slow']) & (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1)), 'Buy',
            np.where((df['EMA_Fast'] < df['EMA_Slow']) & (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1)), 'Sell', None)
        )
        # strength = normalized distance between EMAs
        df['Confidence'] = (abs(df['EMA_Fast'] - df['EMA_Slow']) / df['close']) * 100

        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], name='EMA Fast'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], name='EMA Slow'))

    elif strategy == "RSI":
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['Signal'] = np.where(df['RSI'] < 30, 'Buy', np.where(df['RSI'] > 70, 'Sell', None))
        # strength = distance from neutral 50
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
        # strength = abs(histogram) / avg(|histogram|) normalized to 100
        avg_hist = df['Histogram'].abs().rolling(20).mean()
        df['Confidence'] = (df['Histogram'].abs() / avg_hist) * 50
        df['Confidence'] = df['Confidence'].clip(0, 100)

        fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=signal_line, name='Signal Line'))

    elif strategy == "Bollinger Bands":
        ma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        upper = ma + 2*std
        lower = ma - 2*std
        df['Upper'], df['Lower'], df['MA'] = upper, lower, ma
        df['Signal'] = np.where(df['close'] < lower, 'Buy', np.where(df['close'] > upper, 'Sell', None))
        # strength = distance from band normalized to band width
        band_width = upper - lower
        df['Confidence'] = (abs(df['close'] - ma) / band_width) * 100

        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))
        fig.add_trace(go.Scatter(x=df.index, y=upper, name='Upper Band'))
        fig.add_trace(go.Scatter(x=df.index, y=lower, name='Lower Band'))
        fig.add_trace(go.Scatter(x=df.index, y=ma, name='MA'))

    else:
        df['Confidence'] = 0
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))

    # annotate signals with confidence
    pts = df[df['Signal'].notna()]
    for ts, row in pts.iterrows():
        conf = min(max(row['Confidence'], 0), 100)
        signals.append((ts, row['Signal'], conf))
        fig.add_trace(go.Scatter(
            x=[ts], y=[row['close']], mode='markers+text',
            marker=dict(color='green' if row['Signal']=='Buy' else 'red', size=12,
                        symbol='arrow-up' if row['Signal']=='Buy' else 'arrow-down'),
            text=[f"{row['Signal']} {int(conf)}%"], textposition='top center' if row['Signal']=='Buy' else 'bottom center',
            showlegend=False
        ))

    fig.update_layout(title=f"{strategy} Strategy", xaxis_rangeslider_visible=False, height=600)
    return fig, signals

# -------------------- Backtest --------------------
# run_backtest unchanged
...

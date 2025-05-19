import streamlit as st
import asyncio
import websockets
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from collections import deque
import plotly.graph_objects as go
from streamlit_javascript import st_javascript
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange

# ------------------ CONFIG ------------------
APP_ID = "1089"  # Deriv App ID
TIMEZONE = "Africa/Johannesburg"

# ------------------ STATE ------------------
signal_log = deque(maxlen=100)
candle_df = pd.DataFrame()

# ------------------ STRATEGIES ------------------
strategy_descriptions = {
    "EMA_Cross": "Buy when EMA10 crosses above EMA50. Sell when EMA10 crosses below EMA50.",
    "RSI": "Buy when RSI <30 (oversold). Sell when RSI >70 (overbought).",
    "MACD_Cross": "Buy when MACD line crosses above signal line. Sell when it crosses below.",
    "Bollinger_Bounce": "Buy near lower band bounce. Sell near upper band bounce.",
    "StochRSI": "Buy when Stochastic RSI <20. Sell when >80.",
    "ATR_Breakout": "Buy when price breaks above high + ATR. Sell when below low - ATR.",
    "MA_Ribbon": "Buy when short EMAs above long EMAs. Sell on opposite.",
    "Heikin_Ashi": "Buy on HA candle reversal to green. Sell on reversal to red."
}

def plot_strategy(df, strategy_name):
    signals = []
    close = df['close']
    if strategy_name == "EMA_Cross":
        df['EMA10'] = close.ewm(span=10).mean()
        df['EMA50'] = close.ewm(span=50).mean()
        df['signal'] = np.where(df['EMA10'] > df['EMA50'], 'Buy', 'Sell')
        df['confidence'] = np.abs(df['EMA10'] - df['EMA50'])/close*100

    elif strategy_name == "RSI":
        rsi = RSIIndicator(close, window=14).rsi()
        df['signal'] = np.where(rsi < 30, 'Buy', np.where(rsi > 70, 'Sell', ''))
        df['confidence'] = np.where(df['signal']=='Buy', (30-rsi)/30*100, np.where(df['signal']=='Sell', (rsi-70)/30*100, 0))

    elif strategy_name == "MACD_Cross":
        macd = MACD(close)
        df['macd'] = macd.macd()
        df['signal_line'] = macd.macd_signal()
        df['signal'] = np.where(df['macd'] > df['signal_line'], 'Buy', 'Sell')
        df['confidence'] = np.abs(df['macd'] - df['signal_line'])/close*100

    elif strategy_name == "Bollinger_Bounce":
        bb = BollingerBands(close, window=20, window_dev=2)
        df['lower'] = bb.bollinger_lband()
        df['upper'] = bb.bollinger_uband()
        bounce = (close.shift(1) < df['lower']) & (close > df['lower'])
        bounce |= (close.shift(1) > df['upper']) & (close < df['upper'])
        df['signal'] = np.where(bounce & (close < df['lower']+ (df['upper']-df['lower'])/2), 'Buy',
                         np.where(bounce & (close > df['upper']- (df['upper']-df['lower'])/2), 'Sell', ''))
        df['confidence'] = 100

    elif strategy_name == "StochRSI":
        stoch = StochRSIIndicator(close)
        stoch_k = stoch.stochrsi_k()
        df['signal'] = np.where(stoch_k < 20, 'Buy', np.where(stoch_k > 80, 'Sell', ''))
        df['confidence'] = np.abs(stoch_k-50)/50*100

    elif strategy_name == "ATR_Breakout":
        atr = AverageTrueRange(df['high'], df['low'], close, window=14).average_true_range()
        high_break = close > df['high'].shift(1) + atr
        low_break = close < df['low'].shift(1) - atr
        df['signal'] = np.where(high_break, 'Buy', np.where(low_break, 'Sell', ''))
        df['confidence'] = np.where(df['signal']=='Buy', (close-(df['high'].shift(1)+atr))/atr*100,
                          np.where(df['signal']=='Sell', ((df['low'].shift(1)-atr)-close)/atr*100, 0))

    elif strategy_name == "MA_Ribbon":
        spans = [5,10,20,30,50]
        for s in spans:
            df[f'EMA{s}'] = close.ewm(span=s).mean()
        buy = np.logical_and.reduce([df[f'EMA{spans[i]}'] > df[f'EMA{spans[i+1]}'] for i in range(len(spans)-1)])
        sell = np.logical_and.reduce([df[f'EMA{spans[i]}'] < df[f'EMA{spans[i+1]}'] for i in range(len(spans)-1)])
        df['signal'] = np.where(buy, 'Buy', np.where(sell, 'Sell', ''))
        df['confidence'] = np.std([df[f'EMA{s}'] for s in spans], axis=0)/close*100

    else:  # Heikin Ashi
        ha = pd.DataFrame()
        ha['close'] = (df['open'] + df['high'] + df['low'] + df['close'])/4
        ha['open'] = (df['open'].shift(1)+df['close'].shift(1))/2
        ha['high'] = pd.concat([df['high'], ha['open'], ha['close']], axis=1).max(axis=1)
        ha['low'] = pd.concat([df['low'], ha['open'], ha['close']], axis=1).min(axis=1)
        color = ha['close'] > ha['open']
        reversal = color.shift(1) != color
        df['signal'] = np.where(reversal & color, 'Buy', np.where(reversal & ~color, 'Sell', ''))
        df['confidence'] = 100

    # detect edges
    for i in range(1, len(df)):
        if df['signal'].iloc[i] in ['Buy','Sell'] and df['signal'].iloc[i] != df['signal'].iloc[i-1]:
            ts = df.index[i]
            signals.append((ts, df['signal'].iloc[i], int(df['confidence'].iloc[i]), df['close'].iloc[i]))

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
    # overlays
    if strategy_name == 'EMA_Cross':
        fig.add_trace(go.Scatter(x=df.index,y=df['EMA10'],mode='lines',name='EMA10'))
        fig.add_trace(go.Scatter(x=df.index,y=df['EMA50'],mode='lines',name='EMA50'))
    if strategy_name == 'MACD_Cross':
        fig.add_trace(go.Scatter(x=df.index,y=df['macd'],mode='lines',name='MACD'))
        fig.add_trace(go.Scatter(x=df.index,y=df['signal_line'],mode='lines',name='Signal'))
    if strategy_name in ['Bollinger_Bounce']:
        fig.add_trace(go.Scatter(x=df.index,y=df['upper'],mode='lines',name='Upper'))
        fig.add_trace(go.Scatter(x=df.index,y=df['lower'],mode='lines',name='Lower'))
    if strategy_name == 'MA_Ribbon':
        for s in spans:
            fig.add_trace(go.Scatter(x=df.index,y=df[f'EMA{s}'],mode='lines',name=f'EMA{s}'))

    fig.update_layout(title=f"Strategy: {strategy_name}", xaxis_rangeslider_visible=False)
    return fig, signals

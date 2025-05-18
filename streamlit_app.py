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

# -------------------- Strategy Charting with Signals --------------------
def plot_candlestick_with_indicators(df, strategy):
    fig = go.Figure()
    signals = []
    if df.empty:
        return fig, signals

    # indicator & signal computation
    if strategy == "EMA Crossover":
        df['EMA_Fast'] = df['close'].ewm(span=5).mean()
        df['EMA_Slow'] = df['close'].ewm(span=10).mean()
        df['Signal'] = np.where(
            (df['EMA_Fast'] > df['EMA_Slow']) & (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1)), 'Buy',
            np.where((df['EMA_Fast'] < df['EMA_Slow']) & (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1)), 'Sell', None)
        )
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
        df['Signal'] = np.where(
            (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1)), 'Buy',
            np.where((macd < signal_line) & (macd.shift(1) >= signal_line.shift(1)), 'Sell', None)
        )
        fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=signal_line, name='Signal Line'))

    elif strategy == "Bollinger Bands":
        ma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        upper = ma + 2*std
        lower = ma - 2*std
        df['Upper'], df['Lower'], df['MA'] = upper, lower, ma
        df['Signal'] = np.where(df['close'] < lower, 'Buy', np.where(df['close'] > upper, 'Sell', None))
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))
        fig.add_trace(go.Scatter(x=df.index, y=upper, name='Upper Band'))
        fig.add_trace(go.Scatter(x=df.index, y=lower, name='Lower Band'))
        fig.add_trace(go.Scatter(x=df.index, y=ma, name='MA'))

    else:
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))

    # annotate signals
    pts = df[df['Signal'].notna()]
    for ts, row in pts.iterrows():
        signals.append((ts, row['Signal']))
        fig.add_trace(go.Scatter(
            x=[ts], y=[row['close']], mode='markers+text',
            marker=dict(color='green' if row['Signal']=='Buy' else 'red', size=12,
                        symbol='arrow-up' if row['Signal']=='Buy' else 'arrow-down'),
            text=[row['Signal']], textposition='top center' if row['Signal']=='Buy' else 'bottom center',
            showlegend=False
        ))

    fig.update_layout(title=f"{strategy} Strategy", xaxis_rangeslider_visible=False, height=600)
    return fig, signals

# -------------------- Backtest --------------------
def run_backtest(df, strategy):
    if df.empty:
        return {}
    # reuse signal column
    _, _ = plot_candlestick_with_indicators(df.copy(), strategy)
    signals = df[df['Signal'].notna()]
    trades = []
    for i, (ts, row) in enumerate(signals.iterrows()):
        if i+1 < len(df):
            next_price = df['close'].iloc[df.index.get_loc(ts)+1]
            ret = (next_price - row['close']) / row['close'] if row['Signal']=='Buy' else (row['close'] - next_price)/row['close']
            trades.append(ret)
    returns = pd.Series(trades)
    return {
        'trades': len(returns),
        'win_rate': (returns>0).mean(),
        'total_return': returns.sum()
    }

# -------------------- Data Handling --------------------
def build_candles(ticks, interval='1min'):
    df = pd.DataFrame(ticks)
    df['epoch'] = pd.to_datetime(df['epoch'], unit='s', utc=True)
    df = df.set_index('epoch').tz_convert(TIMEZONE)
    ohlc = df['quote'].resample(interval).ohlc().dropna()
    return ohlc

async def fetch_history(symbol, count=500):
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "style": "ticks"
        }))
        data = json.loads(await ws.recv())
        if 'history' in data:
            return [{'epoch': e, 'quote': p} for e, p in zip(data['history']['times'], data['history']['prices'])]
    return []

async def stream_ticks(symbol, interval):
    global subscribed, candle_df
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
    async with websockets.connect(uri) as ws:
        subscribed = True
        await ws.send(json.dumps({"ticks_subscribe": symbol}))
        while subscribed:
            msg = json.loads(await ws.recv())
            if 'tick' in msg:
                tick_buffer.append({'epoch': msg['tick']['epoch'], 'quote': msg['tick']['quote']})
                if len(tick_buffer)>=10:
                    candle_df = build_candles(tick_buffer, interval)
                    st.session_state['df'] = candle_df
            await asyncio.sleep(0.5)

# -------------------- UI --------------------
st.title("📊 Deriv Live Dashboard + Backtest & Suggestions")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    symbol = st.selectbox("Symbol", ["R_100","R_50","1HZ100V","Volatility 75 (1s)"])
    timeframe = st.selectbox("Timeframe", ["1min","5min"])
    strategy = st.selectbox("Strategy", ["EMA Crossover","RSI","MACD","Bollinger Bands"])
    backtest_btn = st.button("Run Backtest")
    start_btn = st.button("Start Live Stream")
    stop_btn = st.button("Stop Stream")

# Actions
if start_btn:
    history = asyncio.run(fetch_history(symbol))
    if history:
        tick_buffer.extend(history)
        st.session_state['df'] = build_candles(tick_buffer, timeframe)
    asyncio.run(stream_ticks(symbol, timeframe))

if stop_btn:
    subscribed = False
    st.success("Streaming stopped.")

# Chart & Signals
if 'df' in st.session_state and not st.session_state['df'].empty:
    df = st.session_state['df']
    chart, signals = plot_candlestick_with_indicators(df.copy(), strategy)
    st.plotly_chart(chart, use_container_width=True)

    # display recent
    if signals:
        st.markdown("### 📌 Recent Signals (Local Time GMT+2)")
        for ts, sig in signals[-5:]:
            time_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            st.write(f"**{sig}** at {time_str}")

# Backtest results and suggestions
if backtest_btn:
    if 'df' in st.session_state and not st.session_state['df'].empty:
        stats = run_backtest(st.session_state['df'].copy(), strategy)
        st.subheader("Backtest Results")
        st.write(f"• Trades: {stats['trades']}")
        st.write(f"• Win Rate: {stats['win_rate']*100:.1f}%")
        st.write(f"• Total Return: {stats['total_return']*100:.1f}%")
        
        # suggestion
        if stats['win_rate'] > 0.6 and stats['total_return']>0:
            st.success("👍 Strategy looks strong—consider live trading with proper risk management.")
        else:
            st.warning("⚠️ Strategy underperforms in backtest. Adjust parameters or choose another.")
    else:
        st.error("No data to backtest—run the live fetch first.")

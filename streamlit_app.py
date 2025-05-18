import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")

# --- Utility functions ---
@st.cache_data

def convert_ticks_to_df(ticks):
    df = pd.DataFrame(ticks)
    df['epoch'] = pd.to_datetime(df['epoch'], unit='s')
    df = df.rename(columns={'quote': 'close'})
    df.set_index('epoch', inplace=True)
    df['open'] = df['close'].shift(1)
    df['high'] = df['close'].rolling(2).max()
    df['low'] = df['close'].rolling(2).min()
    df['volume'] = 100  # Placeholder
    return df.dropna()

def calculate_indicators(df, strategy):
    if strategy == "MACD":
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
        df['Hist'] = df['MACD'] - df['Signal_Line']
    elif strategy == "RSI":
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
    elif strategy == "Bollinger Bands":
        df['MA'] = df['close'].rolling(window=20).mean()
        df['STD'] = df['close'].rolling(window=20).std()
        df['Upper'] = df['MA'] + (df['STD'] * 2)
        df['Lower'] = df['MA'] - (df['STD'] * 2)
    return df.dropna()

def detect_signals(df, strategy):
    signal, confidence = None, 0
    if strategy == "MACD":
        if df['Hist'].iloc[-1] > 0 and df['Hist'].iloc[-2] < 0:
            signal = "BUY"
            confidence = min(100, abs(df['Hist'].iloc[-1]) * 500)
        elif df['Hist'].iloc[-1] < 0 and df['Hist'].iloc[-2] > 0:
            signal = "SELL"
            confidence = min(100, abs(df['Hist'].iloc[-1]) * 500)
    elif strategy == "RSI":
        rsi = df['RSI'].iloc[-1]
        if rsi < 30:
            signal = "BUY"
            confidence = 100 - rsi
        elif rsi > 70:
            signal = "SELL"
            confidence = rsi - 70
    return signal, round(confidence, 1)

def plot_candlestick_with_indicators(df, strategy):
    fig = go.Figure()
    if strategy != "RSI":
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))
    if strategy == "MACD":
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal'))
    elif strategy == "RSI":
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
    elif strategy == "Bollinger Bands":
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='Upper Band'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='Lower Band'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA'], name='MA'))
    fig.update_layout(title=strategy, xaxis_rangeslider_visible=False, height=700)
    return fig

# --- WebSocket Handler ---
async def deriv_stream(app_id, symbol, granularity):
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    try:
        async with websockets.connect(uri) as websocket:
            st.session_state['streaming'] = True
            ticks_request = json.dumps({
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": 100,
                "granularity": granularity,
                "style": "candles",
                "subscribe": 1
            })
            await websocket.send(ticks_request)

            while st.session_state.get("streaming", False):
                response = await websocket.recv()
                data = json.loads(response)
                if 'candles' in data:
                    df = convert_ticks_to_df(data['candles'])
                    st.session_state['df'] = df
                    st.session_state['last_tick'] = datetime.utcnow().strftime("%H:%M:%S")
                    break
                elif 'tick' in data:
                    tick = data['tick']
                    st.session_state['last_tick'] = datetime.utcfromtimestamp(tick['epoch']).strftime("%H:%M:%S")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Main App ---
st.title("📈 Deriv WebSocket Trading App")

col1, col2, col3 = st.columns(3)
with col1:
    app_id = st.text_input("Deriv App ID", value="1089")
with col2:
    symbol = st.selectbox("Asset", ["R_100", "R_50", "R_25"])
with col3:
    timeframe = st.selectbox("Timeframe", [1, 5], index=0)

strategy = st.selectbox("Strategy", ["MACD", "RSI", "Bollinger Bands"])
start_btn = st.button("▶️ Start Stream")
stop_btn = st.button("⛔ Stop Stream")
fullscreen = st.checkbox("🖥 Fullscreen chart", value=False)

if stop_btn:
    st.session_state['streaming'] = False
    st.warning("Streaming stopped")

if start_btn and app_id:
    asyncio.run(deriv_stream(app_id, symbol, timeframe * 60))

if "df" in st.session_state and not st.session_state['df'].empty:
    df = calculate_indicators(st.session_state['df'], strategy)
    signal, confidence = detect_signals(df, strategy)
    fig = plot_candlestick_with_indicators(df, strategy)
    st.plotly_chart(fig, use_container_width=True, height=900 if fullscreen else 600)

    if signal:
        st.success(f"📢 Trade Signal: **{signal}** | Confidence Score: **{confidence}/100**")

if "last_tick" in st.session_state:
    st.sidebar.markdown(f"🟢 **Streaming**")
    st.sidebar.markdown(f"📈 Last tick: `{st.session_state['last_tick']}`")
else:
    st.sidebar.markdown("🔴 **Not Connected**")

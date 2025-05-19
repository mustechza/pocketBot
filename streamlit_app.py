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
    "RSI">: "Buy when RSI <30 (oversold). Sell when RSI >70 (overbought).",
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
        bounce = close.shift(1) < df['lower']  &  close > df['lower']
        bounce |= close.shift(1) > df['upper'] & close < df['upper']
        df['signal'] = np.where(bounce & (close < df['lower']+ (df['upper']-df['lower'])/2), 'Buy',
                         np.where(bounce & (close > df['upper']- (df['upper']-df['lower'])/2), 'Sell', ''))
        df['confidence'] = 100

    elif strategy_name == "StochRSI":
        stoch = StochRSIIndicator(close)
        stoch_k = stoch.stochrsi_k()
        stoch_d = stoch.stochrsi_d()
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
        buy = all(df[f'EMA{spans[i]}'] > df[f'EMA{spans[i+1]}'] for i in range(len(spans)-1))
        sell = all(df[f'EMA{spans[i]}'] < df[f'EMA{spans[i+1]}'] for i in range(len(spans)-1))
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

    fig.update_layout(title=f"{asset} - {strategy_name}", xaxis_rangeslider_visible=False)
    return fig, signals

# ------------------ ALERT ------------------
def play_alert():
    st_javascript("""
    const audio = new Audio('https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg');
    audio.play();
    Notification.requestPermission().then(function(result) {
        if (result === 'granted') {
            new Notification("📢 New Signal!", { body: "A new trading signal has been generated." });
        }
    });
    """)

# ------------------ WEBSOCKET ------------------
async def deriv_ws_listener(asset, granularity, update_callback):
    url = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
    async with websockets.connect(url) as ws:
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
                df['epoch'] = pd.to_datetime(df['epoch'], unit='s').dt.tz_localize('UTC').dt.tz_convert(TIMEZONE)
                df.set_index('epoch', inplace=True)
                await update_callback(df)

# ------------------ MAIN ------------------
async def main():
    st.set_page_config(page_title="Deriv Signal App", layout="wide")
    st.title("📡 Deriv Live Signal Dashboard")
    st.caption("Streaming real-time signals from Deriv.")

    # Sidebar
    st.sidebar.header("Settings")
    asset = st.sidebar.selectbox("Asset", ["R_100", "R_75", "R_50", "R_25"], index=0)
    st.sidebar.info(strategy_descriptions)
    strategy = st.sidebar.selectbox("Strategy", list(strategy_descriptions.keys()))
    with st.sidebar.expander("ℹ️ Strategy Info"):
        st.write(strategy_descriptions[strategy])

    trade_duration = st.sidebar.number_input("Trade Duration (minutes)", min_value=1, max_value=60, value=2)
    streaming = st.sidebar.toggle("🔄 Live Streaming", value=True)
    filter_type = st.sidebar.selectbox("Filter Signals", ["All", "Buy", "Sell"])
    compare = st.sidebar.checkbox("📊 Compare Strategies")

    gran_map = {"1 Minute": 60, "5 Minutes": 300, "15 Minutes": 900}
    tf = st.sidebar.selectbox("Candle Timeframe", list(gran_map.keys()))
    granularity = gran_map[tf]

    if not streaming:
        st.warning("🔌 Streaming Paused")
        return

    placeholder_chart = st.empty()
    placeholder_metrics = st.empty()
    placeholder_signals = st.empty()

    async def update_callback(df):
        global candle_df
        candle_df = df

        # Plot current strategy
        fig, signals = plot_strategy(candle_df.copy(), strategy)
        placeholder_chart.plotly_chart(fig, use_container_width=True)

        now = pd.Timestamp.utcnow().tz_localize('UTC').tz_convert(TIMEZONE)

        # Evaluate outcomes
        for s in signal_log:
            if s['outcome'] == 'Pending':
                expiry = pd.to_datetime(s['expiry']).tz_convert(TIMEZONE)
                if now >= expiry and not candle_df[candle_df.index >= expiry].empty:
                    final_price = candle_df[candle_df.index >= expiry].iloc[0]['close']
                    s['outcome'] = 'Win' if (s['signal']=='Buy' and final_price > s['price']) or (s['signal']=='Sell' and final_price < s['price']) else 'Loss'

        # Add new signals
        for ts, sig, conf, price in signals[-3:]:
            ts_fmt = ts.strftime('%Y-%m-%d %H:%M:%S')
            if not any(s['timestamp']==ts_fmt and s['signal']==sig for s in signal_log):
                expiry = ts + timedelta(minutes=trade_duration)
                expiry_fmt = expiry.strftime('%Y-%m-%d %H:%M:%S')
                entry = {"signal": sig, "price": price, "timestamp": ts_fmt,
                         "expiry": expiry_fmt, "duration": trade_duration,
                         "confidence": conf, "outcome": 'Pending'}
                signal_log.appendleft(entry)
                play_alert()

        # Display signals
        filtered = [s for s in signal_log if filter_type=='All' or s['signal']==filter_type]
        display = filtered[:3]
        with placeholder_signals.container():
            st.subheader(f"📢 Latest Signals ({filter_type})")
            cols = st.columns(3)
            for i, s in enumerate(display):
                with cols[i]:
                    st.markdown(f"""
                    <div style='border:1px solid #ddd;padding:10px;border-radius:10px;'>
                        <h5>📌 <b>{s['signal']}</b></h5>
                        <p>💵 Price: <code>{s['price']:.2f}</code></p>
                        <p>⏰ Time: <code>{s['timestamp']}</code></p>
                        <p>⏳ Expires: <code>{s['expiry']}</code> ({s['duration']}m)</p>
                        <p>🎯 Confidence: <code>{s['confidence']}%</code></p>
                        <p>✅ Outcome: <b>{s['outcome']}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

        # Metrics
        total = len(signal_log)
        buys = sum(1 for s in signal_log if s['signal']=='Buy')
        sells = sum(1 for s in signal_log if s['signal']=='Sell')
        wins = sum(1 for s in signal_log if s['outcome']=='Win')
        losses = sum(1 for s in signal_log if s['outcome']=='Loss')
        avg_conf = np.mean([s['confidence'] for s in signal_log]) if signal_log else 0
        avg_dur = np.mean([s['duration'] for s in signal_log]) if signal_log else 0
        winrate = (wins/(wins+losses)*100) if (wins+losses)>0 else 0
        profit_per_trade = 0.95
        roi = ((wins*profit_per_trade - losses)/(max(1, wins+losses))*100)
        profit_factor = (wins*profit_per_trade)/max(1, losses)

        with placeholder_metrics.container():
            st.markdown("---")
            st.subheader("📊 Strategy Performance")
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.metric("Total Signals", total)
            c2.metric("Buy/Sell", f"{buys}/{sells}")
            c3.metric("Avg Confidence", f"{avg_conf:.1f}%")
            c4.metric("Avg Duration (m)", f"{avg_dur:.1f}")
            c5.metric("Win Rate", f"{winrate:.1f}%")
            c6.metric("ROI", f"{roi:.1f}%")
            c7.metric("Profit Factor", f"{profit_factor:.2f}")

        # History chart
        if signal_log:
            hist = pd.DataFrame(signal_log)
            hist['timestamp'] = pd.to_datetime(hist['timestamp'])
            hist['result'] = hist['outcome'].map({'Win':1,'Loss':-1,'Pending':0}).fillna(0)
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Scatter(x=hist['timestamp'], y=hist['result'], mode='lines+markers', name='Outcome'))
            st.subheader("📈 Signal History")
            st.plotly_chart(hist_fig, use_container_width=True)

        # Multi-strategy comparison
        if compare and not candle_df.empty:
            metrics = []
            for strat in strategy_descriptions.keys():
                _, sigs = plot_strategy(candle_df.copy(), strat)
                w=l=0
                for ts, sig, conf, price in sigs:
                    exp = ts + timedelta(minutes=trade_duration)
                    future = candle_df[candle_df.index>=exp]
                    if not future.empty:
                        final = future.iloc[0]['close']
                        if (sig=='Buy' and final>price) or (sig=='Sell' and final<price): w+=1
                        else: l+=1
                wr = (w/(w+l)*100) if (w+l)>0 else 0
                metrics.append((strat, len(sigs), w, l, round(wr,1)))
            comp_df = pd.DataFrame(metrics, columns=["Strategy","Signals","Wins","Losses","Win Rate (%)"]).set_index("Strategy")
            st.subheader("🔍 Strategy Comparison")
            st.dataframe(comp_df)

    # Run listener
    await deriv_ws_listener(asset, granularity, update_callback)

if __name__ == "__main__":
    asyncio.run(main())

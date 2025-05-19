import streamlit as st
import asyncio
import websockets
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from collections import deque
import plotly.graph_objects as go

# ------------------ CONFIG ------------------
APP_ID = "1089"  # Deriv App ID
asset = "R_100"
strategy = "EMA_Cross"
TIMEZONE = "Etc/UTC"

signal_log = deque(maxlen=100)
candle_df = pd.DataFrame()

# ------------------ STRATEGY ------------------
def plot_strategy(df, strategy_name):
    signals = []
    if strategy_name == "EMA_Cross":
        df["EMA10"] = df["close"].ewm(span=10).mean()
        df["EMA50"] = df["close"].ewm(span=50).mean()

        df["signal"] = np.where(df["EMA10"] > df["EMA50"], "Buy",
                         np.where(df["EMA10"] < df["EMA50"], "Sell", ""))
        df["confidence"] = np.abs(df["EMA10"] - df["EMA50"]) / df["close"] * 100

        for i in range(1, len(df)):
            if df["signal"].iloc[i] in ["Buy", "Sell"] and df["signal"].iloc[i] != df["signal"].iloc[i-1]:
                timestamp = df.index[i]
                sig = df["signal"].iloc[i]
                conf = df["confidence"].iloc[i]
                price = df["close"].iloc[i]
                signals.append((timestamp, sig, conf, price))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df["open"],
                                 high=df["high"],
                                 low=df["low"],
                                 close=df["close"],
                                 name="Candles"))
    if "EMA10" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA10"], mode='lines', name='EMA10'))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], mode='lines', name='EMA50'))

    fig.update_layout(title=f"{asset} - {strategy_name}", xaxis_rangeslider_visible=False)
    return fig, signals

# ------------------ WEBSOCKET ------------------
async def deriv_ws_listener(update_callback):
    async with websockets.connect(f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}") as ws:
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
                    df['epoch'] = pd.to_datetime(df['epoch'], unit='s')\
                                 .dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
                    df.set_index('epoch', inplace=True)
                    await update_callback(df)
            except Exception as e:
                print("WebSocket Error:", e)
                continue

# ------------------ MAIN ------------------
async def main():
    st.set_page_config(page_title="Deriv Signal App", layout="wide")
    st.title("📡 Deriv Live Signal Dashboard")
    st.caption("Streaming real-time signals from Deriv with EMA crossover strategy.")

    # Sidebar controls
    st.sidebar.header("Settings")
    trade_duration = st.sidebar.number_input(
        "Trade Duration (minutes)", min_value=1, max_value=60, value=2, step=1
    )

    placeholder_chart = st.empty()
    placeholder_metrics = st.empty()
    placeholder_signals = st.empty()

    async def update_callback(df):
        global candle_df
        candle_df = df

        fig, signals = plot_strategy(candle_df.copy(), strategy)
        placeholder_chart.plotly_chart(fig, use_container_width=True)

        now = datetime.utcnow().replace(tzinfo=pd.Timestamp.now().tz)

        # Evaluate expired signals
        for s in signal_log:
            if 'outcome' not in s:
                expiry_dt = pd.to_datetime(s['expiry']).tz_convert(TIMEZONE)
                if now >= expiry_dt:
                    post_expiry_price = df[df.index >= expiry_dt].head(1)["close"]
                    if not post_expiry_price.empty:
                        final_price = post_expiry_price.values[0]
                        if s['signal'] == 'Buy':
                            s['outcome'] = "Win" if final_price > s['price'] else "Loss"
                        elif s['signal'] == 'Sell':
                            s['outcome'] = "Win" if final_price < s['price'] else "Loss"
                    else:
                        s['outcome'] = "Pending"

        # Process new signals
        for ts, sig, conf, price in reversed(signals[-3:]):
            ts_fmt = ts.strftime('%Y-%m-%d %H:%M:%S')
            expiry = ts + timedelta(minutes=trade_duration)
            expiry_fmt = expiry.strftime('%Y-%m-%d %H:%M:%S')
            if not any(s['timestamp'] == ts_fmt and s['signal'] == sig for s in signal_log):
                signal_log.appendleft({
                    "signal": sig,
                    "price": price,
                    "timestamp": ts_fmt,
                    "expiry": expiry_fmt,
                    "duration": trade_duration,
                    "confidence": int(conf),
                    "outcome": "Pending"
                })

        # Display latest signals
        with placeholder_signals.container():
            st.subheader(f"📢 Latest Signals (Duration: {trade_duration}m)")
            cols = st.columns(3)
            for i, sig in enumerate(list(signal_log)[:3]):
                with cols[i]:
                    st.markdown(f"""
                        <div style="border:1px solid #ddd;padding:10px;border-radius:10px;">
                            <h5 style="margin-bottom:5px;">📌 <b>{sig['signal']}</b></h5>
                            <p>💵 Price: <code>{sig['price']:.2f}</code></p>
                            <p>🕒 Time: <code>{sig['timestamp']}</code></p>
                            <p>⏳ Expires: <code>{sig['expiry']}</code> ({sig['duration']}m)</p>
                            <p>🎯 Confidence: <code>{sig['confidence']}%</code></p>
                            <p>✅ Outcome: <b>{sig['outcome']}</b></p>
                        </div>
                    """, unsafe_allow_html=True)

        # Performance metrics
        with placeholder_metrics.container():
            st.markdown("---")
            st.subheader("📊 Strategy Performance")
            total_signals = len(signal_log)
            buys = sum(1 for s in signal_log if s["signal"] == "Buy")
            sells = sum(1 for s in signal_log if s["signal"] == "Sell")
            avg_conf = np.mean([s["confidence"] for s in signal_log]) if signal_log else 0
            avg_duration = np.mean([s["duration"] for s in signal_log]) if signal_log else 0
            wins = sum(1 for s in signal_log if s.get("outcome") == "Win")
            losses = sum(1 for s in signal_log if s.get("outcome") == "Loss")
            winrate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("📈 Total Signals", total_signals)
            col2.metric("🟢 Buy / 🔴 Sell", f"{buys} / {sells}")
            col3.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")
            col4.metric("⏱ Avg Duration (m)", f"{avg_duration:.1f}")
            col5.metric("🏆 Win Rate", f"{winrate:.1f}%")

    await deriv_ws_listener(update_callback)

if __name__ == "__main__":
    asyncio.run(main())

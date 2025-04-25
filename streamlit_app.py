import streamlit as st
import pandas as pd
import requests
import numpy as np
import mplfinance as mpf
from io import BytesIO
from datetime import datetime
import time

# === Configuration ===
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]
INTERVALS = ["1m", "5m", "15m"]
LIMIT = 100
TOP_N = 3

# === Telegram ===
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=data)

def send_chart_to_telegram(image_buf, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {'photo': image_buf}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}
    requests.post(url, files=files, data=data)

# === Technical Indicators ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(df):
    df["rsi"] = compute_rsi(df["close"], 14)
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["signal"] = df["macd"].ewm(span=9).mean()
    return df

# === Candlestick Pattern ===
def detect_candle_pattern(df):
    if len(df) < 2:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    if last['close'] > last['open'] and prev['close'] < prev['open'] and last['open'] < prev['close'] and last['close'] > prev['open']:
        return "Bullish Engulfing"
    if last['close'] < last['open'] and prev['close'] > prev['open'] and last['open'] > prev['close'] and last['close'] < prev['open']:
        return "Bearish Engulfing"
    return None

# === Fetch Klines ===
def fetch_klines(symbol, interval):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={LIMIT}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    )
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df[['time', 'open', 'high', 'low', 'close', 'volume']]

# === Candlestick Chart ===
def generate_chart(df, symbol):
    df = df.set_index("time").tail(30)
    buf = BytesIO()
    mpf.plot(df, type='candle', style='charles', volume=False,
             title=f"{symbol} - Last 30 Candles", ylabel='Price',
             savefig=dict(fname=buf, dpi=150, format='png'))
    buf.seek(0)
    return buf

# === Signal Evaluation ===
def evaluate_signals(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    score = 0
    signals = []

    if last['rsi'] < 30:
        signals.append("RSI Oversold")
        score += 2
    elif last['rsi'] > 70:
        signals.append("RSI Overbought")
        score += 2

    if last['macd'] > last['signal'] and prev['macd'] < prev['signal']:
        signals.append("MACD Bullish Cross")
        score += 2
    elif last['macd'] < last['signal'] and prev['macd'] > prev['signal']:
        signals.append("MACD Bearish Cross")
        score += 2

    pattern = detect_candle_pattern(df)
    if pattern:
        signals.append(pattern)
        score += 1

    return signals, score

# === Multi-interval Analysis ===
def analyze_symbol(symbol):
    combined_score = 0
    all_signals = []
    latest_df = None

    for interval in INTERVALS:
        df = fetch_klines(symbol, interval)
        df = calculate_indicators(df)
        signals, score = evaluate_signals(df)
        combined_score += score
        all_signals.extend([f"[{interval}] {sig}" for sig in signals])
        if interval == "1m":
            latest_df = df

    return combined_score, all_signals, latest_df

# === Streamlit UI ===
st.set_page_config(page_title="Pocket Option Signal Bot", layout="wide")
st.title("📈 Pocket Option Signal Dashboard")
st.markdown("---")

selected_assets = st.multiselect("Choose Assets", SYMBOLS, default=SYMBOLS)
refresh = st.button("🔄 Refresh Now")

if refresh:
    results = []
    for symbol in selected_assets:
        score, signals, df = analyze_symbol(symbol)
        results.append({"symbol": symbol, "score": score, "signals": signals, "df": df})

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    cols = st.columns(len(sorted_results))
    for i, r in enumerate(sorted_results):
        with cols[i]:
            st.subheader(f"{r['symbol']}")
            st.metric("Confidence", r["score"])
            for sig in r["signals"]:
                st.markdown(f"- {sig}")

    st.markdown("---")
    for r in sorted_results[:TOP_N]:
        chart_img = generate_chart(r["df"], r["symbol"])
        st.image(chart_img, caption=f"{r['symbol']} Chart")

        msg = f"📊 {r['symbol']} Signal(s):\n" + "\n".join(f"• {s}" for s in r['signals'])
        send_chart_to_telegram(chart_img, msg)
        send_telegram_message(msg)

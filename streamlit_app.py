import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import mplfinance as mpf
from io import BytesIO

# === Configuration ===
TELEGRAM_BOT_TOKEN = "8037778857:AAFavL2gBXOKbJnoUOoFk1vApDlBw1Lc5rs"
TELEGRAM_CHAT_ID = "6722676136"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]  # Add more Binance symbols here
INTERVAL = "1m"
LIMIT = 100
SLEEP_TIME = 60  # seconds between checks

# === Telegram Functions ===
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print(f"Telegram Error: {e}")

def send_chart_to_telegram(image_buf, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {'photo': image_buf}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}
    try:
        requests.post(url, files=files, data=data)
    except Exception as e:
        print(f"Telegram Image Error: {e}")

# === Indicator Calculations ===
def calculate_indicators(df):
    df["rsi"] = compute_rsi(df["close"], 14)
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["signal"] = df["macd"].ewm(span=9).mean()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# === Candlestick Chart Generator ===
def generate_candlestick_chart(df, symbol):
    df_chart = df.copy().set_index('time').tail(30)
    df_chart["open"] = df_chart["open"].astype(float)
    df_chart["high"] = df_chart["high"].astype(float)
    df_chart["low"] = df_chart["low"].astype(float)
    df_chart["close"] = df_chart["close"].astype(float)
    
    buf = BytesIO()
    mpf.plot(df_chart, type='candle', style='charles', volume=False,
             title=f"{symbol} - Last 30 Candles", ylabel='Price',
             savefig=dict(fname=buf, dpi=150, format='png'))
    buf.seek(0)
    return buf

# === Candlestick Pattern Logic (Simple) ===
def detect_candle_pattern(df):
    if len(df) < 2:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Bullish Engulfing
    if last['close'] > last['open'] and prev['close'] < prev['open'] and last['open'] < prev['close'] and last['close'] > prev['open']:
        return "Bullish Engulfing"
    # Bearish Engulfing
    if last['close'] < last['open'] and prev['close'] > prev['open'] and last['open'] > prev['close'] and last['close'] < prev['open']:
        return "Bearish Engulfing"
    return None

# === Binance Price Data ===
def fetch_klines(symbol, interval, limit):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df[['time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"⚠️ Error fetching {symbol} data: {e}")
        return None

# === Strategy Evaluation ===
def check_signals(df):
    if len(df) < 2:
        return []

    last = df.iloc[-1]
    signals = []

    # RSI Overbought/Oversold
    if last['rsi'] > 70:
        signals.append("RSI Overbought")
    elif last['rsi'] < 30:
        signals.append("RSI Oversold")

    # MACD crossover
    if last['macd'] > last['signal'] and df.iloc[-2]['macd'] < df.iloc[-2]['signal']:
        signals.append("MACD Bullish Cross")
    elif last['macd'] < last['signal'] and df.iloc[-2]['macd'] > df.iloc[-2]['signal']:
        signals.append("MACD Bearish Cross")

    # Candle pattern
    pattern = detect_candle_pattern(df)
    if pattern:
        signals.append(pattern)

    return signals

# === Main Loop ===
def run_bot():
    while True:
        for symbol in SYMBOLS:
            df = fetch_klines(symbol, INTERVAL, LIMIT)
            if df is not None and len(df) > 0:
                df = calculate_indicators(df)
                signals = check_signals(df)

                if signals:
                    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                    msg = f"📊 [{timestamp}] {symbol} Signal(s):\n" + "\n".join(f"• {s}" for s in signals)
                    print(msg)
                    chart = generate_candlestick_chart(df, symbol)
                    send_chart_to_telegram(chart, msg)

        time.sleep(SLEEP_TIME)

# === Run ===
if __name__ == "__main__":
    run_bot()

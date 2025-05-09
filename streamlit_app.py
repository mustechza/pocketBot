import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import datetime
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components

# --- SETTINGS ---
REFRESH_INTERVAL = 10  # seconds
CANDLE_LIMIT = 500
BYBIT_URL = "https://api.bybit.com/v5/market/kline"
ASSETS = ["ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]

# --- AUTO REFRESH ---
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh")

# --- SIDEBAR: Parameter Inputs ---
uploaded_file = st.sidebar.file_uploader("Upload historical data (CSV)", type=["csv"])
selected_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:2])
selected_strategy = st.sidebar.selectbox("Strategy", [
    "EMA Cross", "RSI Divergence", "MACD Cross", "Bollinger Band Bounce",
    "Stochastic Oscillator", "EMA + RSI Combined", "ML Model (Random Forest)"
])
money_strategy = st.sidebar.selectbox("Money Management", ["Flat", "Martingale"])

# Indicator parameter inputs
ema_short = st.sidebar.number_input("EMA Short Period", 2, 50, value=5)
ema_long = st.sidebar.number_input("EMA Long Period", 5, 100, value=20)
rsi_period = st.sidebar.number_input("RSI Period", 5, 50, value=14)
stoch_period = st.sidebar.number_input("Stochastic Period", 5, 50, value=14)
bb_period = st.sidebar.number_input("Bollinger Band Period", 5, 50, value=20)

# --- FUNCTIONS ---
def fetch_candles(symbol, interval='1', limit=500, category='linear'):
    """
    Fetches historical candlestick data from Bybit.

    Parameters:
    - symbol (str): Trading pair symbol, e.g., 'ETHUSDT'
    - interval (str): Kline interval. Options include '1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M'
    - limit (int): Number of data points to retrieve (max 1000)
    - category (str): Market category. Options include 'spot', 'linear', 'inverse'

    Returns:
    - pd.DataFrame: DataFrame containing candlestick data
    """
    try:
        endpoint = BYBIT_URL
        params = {
            'category': category,
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data['retCode'] != 0:
            raise ValueError(f"API Error: {data['retMsg']}")

        # Extract kline data
        kline_data = data['result']['list']

        # Create DataFrame
        df = pd.DataFrame(kline_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        # Convert data types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        return df

    except Exception as e:
        st.warning(f"Fetching data failed: {e}")
        return None

def calculate_indicators(df):
    df['EMA5'] = df['close'].ewm(span=ema_short, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=ema_long, adjust=False).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['BB_upper'] = df['close'].rolling(window=bb_period).mean() + 2 * df['close'].rolling(window=bb_period).std()
    df['BB_lower'] = df['close'].rolling(window=bb_period).mean() - 2 * df['close'].rolling(window=bb_period).std()

    low_min = df['low'].rolling(window=stoch_period).min()
    high_max = df['high'].rolling(window=stoch_period).max()
    df['Stochastic'] = (df['close'] - low_min) / (high_max - low_min) * 100

    return df

def generate_signal(timestamp, signal_type, price):
    duration = 5
    return {
        "Time": timestamp,
        "Signal": signal_type,
        "Price": price,
        "Trade Duration (min)": duration
    }

def detect_signals(df, strategy):
    signals = []
    for i in range(1, len(df)):
        t = df['timestamp'].iloc[i]
        price = df['close'].iloc[i]

        if strategy == "EMA Cross":
            if df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i]:
                signals.append(generate_signal(t, "Buy (EMA Cross)", price))
            elif df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i]:
                signals.append(generate_signal(t, "Sell (EMA Cross)", price))

        elif strategy == "RSI Divergence":
            rsi = df['RSI'].iloc[i]
            if rsi < 30:
                signals.append(generate_signal(t, "Buy (RSI Oversold)", price))
            elif rsi > 70:
                signals.append(generate_signal(t, "Sell (RSI Overbought)", price))

        elif strategy == "MACD Cross":
            if df['MACD'].iloc[i-1] < df['MACD_signal'].iloc[i-1] and df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                signals.append(generate_signal(t, "Buy (MACD Cross)", price))
            elif df['MACD'].iloc[i-1] > df['MACD_signal'].iloc[i-1] and df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
                signals.append(generate_signal(t, "Sell (MACD Cross)", price))

        elif strategy == "Bollinger Band Bounce":
            if df['close'].iloc[i] < df['BB_lower'].iloc[i]:
                signals.append(generate_signal(t, "Buy (BB Lower)", price))
            elif df['close'].iloc[i] > df['BB_upper'].iloc[i]:
                signals.append(generate_signal(t, "Sell (BB Upper)", price))

        elif strategy == "Stochastic Oscillator":
            stoch = df['Stochastic'].iloc[i]
            if stoch < 20:
                signals.append(generate_signal(t, "Buy (Stochastic)", price))
            elif stoch > 80:
                signals.append(generate_signal(t, "Sell (Stochastic)", price))

        elif strategy == "EMA + RSI Combined":
            if df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i] and df['RSI'].iloc[i] < 40:
                signals.append(generate_signal(t, "Buy (EMA+RSI)", price))
            elif df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i] and df['RSI'].iloc[i] > 60:
                signals.append(generate_signal(t, "Sell (EMA+RSI)", price))
    return signals

def train_ml_model(df):
    df = df.copy()
    df['target'] = (df['close'].shift(-2) > df['close']).astype(int)
    features = ['EMA5', 'EMA20', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Stochastic']
    df = df.dropna(subset=features + ['target'])

    X = df[features]
    y = df['target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    df['ML_Prediction'] = model.predict(X)
    df['Signal'] = df['ML_Prediction'].map({1: 'Buy (ML)', 0: 'Sell (ML)'})
    df['Trade Duration (min)'] = 2

    return df[['timestamp', 'Signal', 'close', 'Trade Duration (min)']].rename(columns={'close': 'Price'})

def simulate_money_management(signals, strategy="Flat", initial_balance=1000, bet_size=10):
    balance = initial_balance
    last_bet = bet_size
    wins, losses, pnl = 0, 0, []

    result_log = []
    for s in signals:
        win = np.random.choice([True, False], p=[0.55, 0.45])
        if win:
            balance += last_bet
            result = "Win"
            wins += 1
            last_bet = bet_size
        else:
            balance -= last_bet
            result = "Loss"
            losses += 1
            if strategy == "Martingale":
                last_bet *= 2
        pnl.append(balance)
        result_log.append({
            "Time": s["Time"], "
::contentReference[oaicite:0]{index=0}
 

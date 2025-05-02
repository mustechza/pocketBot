
import streamlit as st
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
BINANCE_URL = "https://api.binance.com/api/v3/klines"
ASSETS = ["ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]

st.set_page_config(layout="wide")
st.title("📈 Pocket Option Signals | Live + Backtest + Money Management")

# --- AUTO REFRESH ---
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh")

# --- SIDEBAR ---
uploaded_file = st.sidebar.file_uploader("Upload historical data (CSV)", type=["csv"])
selected_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:2])
selected_strategy = st.sidebar.selectbox("Strategy", [
    "EMA Cross", "RSI Divergence", "MACD Cross", "Bollinger Band Bounce",
    "Stochastic Oscillator", "EMA + RSI Combined", "Support/Resistance Breakout", "ML Model (Random Forest)"
])
money_strategy = st.sidebar.selectbox("Money Management", ["Flat", "Martingale"])

# Parameters
ema_short = st.sidebar.number_input("EMA Short Period", 2, 50, value=5)
ema_long = st.sidebar.number_input("EMA Long Period", 5, 100, value=20)
rsi_period = st.sidebar.number_input("RSI Period", 5, 50, value=14)
stoch_period = st.sidebar.number_input("Stochastic Period", 5, 50, value=14)
bb_period = st.sidebar.number_input("Bollinger Band Period", 5, 50, value=20)

# --- FUNCTIONS ---
def fetch_candles(symbol, interval="1m", limit=500):
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        response = requests.get(BINANCE_URL, params=params, timeout=10)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
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

# -- SUPPORT/RESISTANCE breakout signal generation --
def simulate_sr_breakout_trades(df, sr_length=15, sr_margin=2, atr_period=17, tp_pct=0.02, sl_pct=0.01):
    df = df.copy()
    df['ATR'] = df['high'].sub(df['low']).rolling(window=atr_period).mean()
    pivot_highs = df['high'][(df['high'].shift(sr_length) < df['high']) & (df['high'].shift(-sr_length) < df['high'])]
    pivot_lows = df['low'][(df['low'].shift(sr_length) < df['low']) & (df['low'].shift(-sr_length) < df['low'])]
    df['resistance'] = np.nan
    df['support'] = np.nan
    zone_range = (df['high'].max() - df['low'].min()) / df['high'].max()
    for i in range(sr_length, len(df) - sr_length):
        idx = df.index[i]
        if not np.isnan(pivot_highs.iloc[i]):
            top = pivot_highs.iloc[i]
            df.loc[df.index >= idx, 'resistance'] = top
        if not np.isnan(pivot_lows.iloc[i]):
            bottom = pivot_lows.iloc[i]
            df.loc[df.index >= idx, 'support'] = bottom
    df['bull_breakout'] = (df['close'] > df['resistance'].shift(1)) & (df['close'].shift(1) <= df['resistance'].shift(1))
    df['bear_breakout'] = (df['close'] < df['support'].shift(1)) & (df['close'].shift(1) >= df['support'].shift(1))
    trades = []
    position = None
    for i in range(1, len(df)):
        row = df.iloc[i]
        time = df['timestamp'].iloc[i]
        price = row['close']
        if position:
            entry = position['entry_price']
            if position['type'] == 'long':
                if price >= entry * (1 + tp_pct) or price <= entry * (1 - sl_pct):
                    duration = (time - position['entry_time']).total_seconds() / 60
                    trades.append({"Time": position['entry_time'], "Signal": "Buy (Breakout)", "Price": entry, "Trade Duration (min)": duration})
                    position = None
            else:
                if price <= entry * (1 - tp_pct) or price >= entry * (1 + sl_pct):
                    duration = (time - position['entry_time']).total_seconds() / 60
                    trades.append({"Time": position['entry_time'], "Signal": "Sell (Breakdown)", "Price": entry, "Trade Duration (min)": duration})
                    position = None
        if position is None:
            if row['bull_breakout']:
                position = {'type': 'long', 'entry_price': price, 'entry_time': time}
            elif row['bear_breakout']:
                position = {'type': 'short', 'entry_price': price, 'entry_time': time}
    return trades, df

def track_zone_persistence(df):
    zones = []
    active_res, active_sup = None, None
    for i in range(1, len(df)):
        t = df['timestamp'].iloc[i]
        price = df['close'].iloc[i]
        r = df['resistance'].iloc[i]
        s = df['support'].iloc[i]
        if not np.isnan(r) and active_res is None:
            active_res = {'type': 'resistance', 'value': r, 'start_time': t}
        elif active_res and price > active_res['value']:
            persistence = (t - active_res['start_time']).total_seconds() / 60
            zones.append({"Zone Type": "resistance", "Value": active_res['value'], "Start Time": active_res['start_time'], "Break Time": t, "Persistence (min)": persistence})
            active_res = None
        if not np.isnan(s) and active_sup is None:
            active_sup = {'type': 'support', 'value': s, 'start_time': t}
        elif active_sup and price < active_sup['value']:
            persistence = (t - active_sup['start_time']).total_seconds() / 60
            zones.append({"Zone Type": "support", "Value": active_sup['value'], "Start Time": active_sup['start_time'], "Break Time": t, "Persistence (min)": persistence})
            active_sup = None
    return pd.DataFrame(zones)

def plot_sr_zones(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
    if 'resistance' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['resistance'], name='Resistance', line=dict(color='red', dash='dash')))
    if 'support' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['support'], name='Support', line=dict(color='green', dash='dash')))
    fig.update_layout(title=f"{asset} – Support/Resistance Zones", xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=False)
    return fig


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
            "Time": s["Time"], "Signal": s["Signal"], "Result": result,
            "Balance": balance, "Trade Duration (min)": s["Trade Duration (min)"]
        })

    df = pd.DataFrame(result_log)
    max_drawdown = ((df['Balance'].cummax() - df['Balance']) / df['Balance'].cummax()).max()
    roi = ((balance - initial_balance) / initial_balance) * 100
    profit_factor = (wins * bet_size) / max(losses * bet_size, 1)

    return df, {
        "Win Rate (%)": 100 * wins / (wins + losses),
        "ROI (%)": roi,
        "Max Drawdown (%)": 100 * max_drawdown,
        "Profit Factor": profit_factor
    }

def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA5'], name="EMA5", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA20'], name="EMA20", line=dict(color='red')))
    fig.update_layout(title=asset, xaxis_rangeslider_visible=False)
    return fig

def show_browser_alert(message):
    js_code = f"""
    <script>
    alert("{message}");
    </script>
    """
    components.html(js_code)



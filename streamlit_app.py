# crypto_signals_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
import streamlit.components.v1 as components
from datetime import datetime
from binance.client import Client
import time

# SECTION 1: Sidebar Inputs
st.sidebar.title("⚙️ Settings")

data_source = st.sidebar.selectbox("Select Data Source", ["Live (Binance)", "Upload CSV"])
asset = st.sidebar.text_input("Asset (e.g. BTCUSDT)", value="BTCUSDT")
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "1h", "4h", "1d"])
strategy = st.sidebar.selectbox("Strategy", ["EMA Cross", "RSI Divergence", "MACD Cross", 
                                             "Bollinger Band Bounce", "Stochastic Oscillator",
                                             "EMA + RSI Combined", "Machine Learning"])

st.sidebar.markdown("### 📊 Indicator Parameters")
ema_short = st.sidebar.slider("EMA Short", 3, 20, 5)
ema_long = st.sidebar.slider("EMA Long", 10, 50, 20)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
macd_fast = st.sidebar.slider("MACD Fast", 5, 20, 12)
macd_slow = st.sidebar.slider("MACD Slow", 10, 40, 26)
macd_signal = st.sidebar.slider("MACD Signal", 5, 20, 9)
bb_period = st.sidebar.slider("BB Period", 10, 40, 20)

st.sidebar.markdown("### 💰 Money Management")
money_mgmt = st.sidebar.selectbox("Strategy", ["Flat", "Martingale"])
initial_balance = st.sidebar.number_input("Initial Balance ($)", value=1000)
bet_size = st.sidebar.number_input("Bet Size ($)", value=10)

signal_alerts = st.sidebar.checkbox("🔔 Enable Browser Alerts", value=True)


# SECTION 2: Helper Functions
def fetch_candles(symbol, interval=interval, limit=200):
    client = Client()
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def generate_signal(time, signal, price):
    return {
        "Time": time,
        "Signal": signal,
        "Price": round(price, 4),
        "Trade Duration (min)": 2
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


# SECTION 3: Main Dashboard Logic
st.title("📈 Crypto Trading Signals Dashboard")

# Load data
if data_source == "Live (Binance)":
    df = fetch_candles(asset)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

# Calculate indicators
df['EMA5'] = ta.ema(df['close'], length=ema_short)
df['EMA20'] = ta.ema(df['close'], length=ema_long)
df['RSI'] = ta.rsi(df['close'], length=rsi_period)
macd = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
df['MACD'] = macd['MACD_12_26_9']
df['MACD_signal'] = macd['MACDs_12_26_9']
bb = ta.bbands(df['close'], length=bb_period)
df['BB_upper'] = bb['BBU_20_2.0']
df['BB_lower'] = bb['BBL_20_2.0']
stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
df['Stochastic'] = stoch['STOCHk_14_3_3']

# Detect signals
if strategy == "Machine Learning":
    signal_df = train_ml_model(df)
    signals = signal_df.to_dict("records")
else:
    signal_list = detect_signals(df, strategy)
    signals = signal_list

# Show alerts for new signals (latest only)
if signal_alerts and signals:
    latest_signal = signals[-1]
    show_browser_alert(f"{latest_signal['Signal']} at {latest_signal['Time']} | Price: {latest_signal['Price']}")

# Display latest signals
st.subheader("📌 Latest Signals")
if signals:
    signal_df = pd.DataFrame(signals).sort_values("Time", ascending=False).reset_index(drop=True)
    st.dataframe(signal_df.head(10), use_container_width=True)
else:
    st.info("No signals detected.")

# Plot chart
st.subheader("📊 Chart")
st.plotly_chart(plot_chart(df, asset), use_container_width=True)

# Simulate money management
st.subheader("💰 Money Management Simulation")
if signals:
    simulation_df, metrics = simulate_money_management(signals, strategy=money_mgmt,
                                                       initial_balance=initial_balance,
                                                       bet_size=bet_size)
    st.dataframe(simulation_df.tail(10), use_container_width=True)

    st.markdown("### 📉 Performance Metrics")
    st.metric("Win Rate (%)", f"{metrics['Win Rate (%)']:.2f}")
    st.metric("ROI (%)", f"{metrics['ROI (%)']:.2f}")
    st.metric("Max Drawdown (%)", f"{metrics['Max Drawdown (%)']:.2f}")
    st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
else:
    st.warning("No signals to simulate.")

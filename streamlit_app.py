import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier

# -------------------- HELPER FUNCTIONS --------------------
def calculate_indicators(df):
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean() / df['close'].pct_change().rolling(14).std()))
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_upper'] = df['close'].rolling(window=20).mean() + (df['close'].rolling(window=20).std() * 2)
    df['BB_lower'] = df['close'].rolling(window=20).mean() - (df['close'].rolling(window=20).std() * 2)
    df['Stochastic'] = ((df['close'] - df['low'].rolling(14).min()) /
                       (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
    return df.dropna()

def generate_signal(t, signal_type, price):
    return {"Time": t, "Signal": signal_type, "Price": price, "Trade Duration (min)": 2}

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

def show_browser_alert(message):
    js_code = f"""
    <script>
    alert("{message}");
    </script>
    """
    components.html(js_code)

# -------------------- STREAMLIT UI --------------------
st.set_page_config(layout="wide")
st.title("📈 Multi-Strategy Crypto Signal App")

uploaded_file = st.file_uploader("Upload OHLCV CSV (with timestamp, open, high, low, close)")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
    df = df.sort_values('timestamp')

    money_strategy = st.sidebar.selectbox("Money Management Strategy", ["Flat", "Martingale"])

    selected_strategies = st.sidebar.multiselect(
        "Select Strategies for Comparison",
        [
            "EMA Cross", "RSI Divergence", "MACD Cross", "Bollinger Band Bounce",
            "Stochastic Oscillator", "EMA + RSI Combined", "ML Model (Random Forest)"
        ],
        default=["EMA Cross", "RSI Divergence"]
    )

    comparison_results = []

    for strategy in selected_strategies:
        if strategy == "ML Model (Random Forest)":
            df_ind = calculate_indicators(df)
            df_signals = train_ml_model(df_ind)
            signals = df_signals.to_dict('records')
        else:
            df_ind = calculate_indicators(df)
            signals = detect_signals(df_ind, strategy)

        mm_df, stats = simulate_money_management(signals, strategy=money_strategy)
        comparison_results.append({"Strategy": strategy, **stats})

    if comparison_results:
        st.subheader("📊 Strategy Comparison Results")
        st.dataframe(pd.DataFrame(comparison_results))

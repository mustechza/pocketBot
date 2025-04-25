import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# App Configuration
CSV_URL = "https://raw.githubusercontent.com/mustechza/mustech-predict/main/training_data_mock.csv"
THRESHOLD_ALERT = 2.0  # Define a "safe" multiplier threshold

# Load Training Data
@st.cache_data
def load_training_data(url):
    df = pd.read_csv(url)
    df.columns = [c.lower().strip() for c in df.columns]
    if 'target' not in df.columns:
        st.error("CSV must contain a 'target' column.")
        return None, None
    X = df.drop(columns=['target'])
    y = df['target'].apply(lambda x: min(x, 10.5))  # Cap values
    return X, y

X_train, y_train = load_training_data(CSV_URL)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# App Title
st.title("🎯 Crash Predictor + Money Manager")

# Input + Prediction
st.header("📥 Input")
with st.form("input_form"):
    recent_input = st.text_input("Enter recent crash multipliers (comma-separated)")
    feedback_value = st.text_input("Actual next multiplier (optional)")
    strategy = st.selectbox("💸 Choose Strategy", ["Flat Betting", "Martingale", "Anti-Martingale"])
    bankroll = st.number_input("💰 Starting Bankroll", value=100.0)
    base_bet = st.number_input("🎯 Base Bet Amount", value=1.0)
    risk_threshold = st.slider("⚠️ Risk Threshold (stop if bankroll below)", min_value=0.0, max_value=100.0, value=10.0)
    submitted = st.form_submit_button("🔁 Submit")

def parse_input(text):
    try:
        raw = [float(x.strip().lower().replace('x', '')) for x in text.split(',') if x.strip()]
        capped = [min(x, 10.5) for x in raw]
        return capped
    except:
        return []

def extract_features(values):
    if len(values) < 10:
        return None
    last_vals = values[-10:]
    return np.array([[np.mean(last_vals), np.std(last_vals), last_vals[-1],
                      max(last_vals), min(last_vals),
                      last_vals[-1] - last_vals[-2] if len(last_vals) > 1 else 0]])

crash_values = parse_input(recent_input)
features = extract_features(crash_values) if crash_values else None

if features is not None:
    prediction = model.predict(features)[0]
    safe_target = round(prediction * 0.97, 2)
    st.subheader(f"🎯 Predicted: {prediction:.2f}")
    st.success(f"🛡️ Safe Multiplier Target (3% edge): {safe_target:.2f}")

    # Threshold Alert
    if prediction < THRESHOLD_ALERT:
        st.warning(f"⚠️ Alert: Prediction below safety threshold ({THRESHOLD_ALERT}x)")

    # Feedback Handling
    if submitted and feedback_value:
        try:
            feedback = min(float(feedback_value), 10.5)
            X_train = pd.concat([X_train, pd.DataFrame(features, columns=X_train.columns)], ignore_index=True)
            y_train = pd.concat([y_train, pd.Series([feedback])], ignore_index=True)
            model.fit(X_train, y_train)
            st.success("Model retrained with feedback!")

            # Append new multiplier to recent input
            crash_values.append(feedback)
        except:
            st.error("Invalid feedback value.")

# Win/Loss Tracker
if len(X_train) >= 20:
    st.header("📈 Prediction Accuracy (Last 20)")
    preds = model.predict(X_train.tail(20))
    actuals = y_train.tail(20).values
    outcomes = (preds.round(2) <= actuals.round(2))
    result_df = pd.DataFrame({
        "Predicted": preds.round(2),
        "Actual": actuals.round(2),
        "Result": ["✅ Win" if x else "❌ Loss" for x in outcomes]
    })
    st.dataframe(result_df.style.applymap(
        lambda val: 'background-color: #d4edda' if val == "✅ Win" else 'background-color: #f8d7da',
        subset=["Result"]
    ))

    # Summary Stats
    win_rate = np.mean(outcomes)
    st.metric("🏆 Win Rate", f"{win_rate*100:.1f}%")
    st.metric("✅ Wins", int(np.sum(outcomes)))
    st.metric("❌ Losses", int(np.sum(~outcomes)))

# ================================
# 💸 Money Management Simulation
# ================================
if submitted and features is not None:
    st.header("💸 Strategy Simulation")
    simulated_bankroll = bankroll
    current_bet = base_bet
    logs = []

    for pred, actual in zip(preds, actuals):
        win = pred <= actual
        logs.append({
            "Bankroll Before": simulated_bankroll,
            "Bet": current_bet,
            "Prediction": round(pred, 2),
            "Actual": round(actual, 2),
            "Win": win
        })
        if win:
            simulated_bankroll += current_bet
            if strategy == "Anti-Martingale":
                current_bet *= 2
            else:
                current_bet = base_bet
        else:
            simulated_bankroll -= current_bet
            if strategy == "Martingale":
                current_bet *= 2
            else:
                current_bet = base_bet

        # Stop if risk threshold crossed
        if simulated_bankroll < risk_threshold:
            break

    log_df = pd.DataFrame(logs)
    st.dataframe(log_df.style.applymap(
        lambda v: 'background-color: #d4edda' if v is True else 'background-color: #f8d7da',
        subset=["Win"]
    ))

    st.subheader("💹 Final Result")
    st.metric("Final Bankroll", f"{simulated_bankroll:.2f}")
    st.metric("Profit/Loss", f"{simulated_bankroll - bankroll:.2f}")

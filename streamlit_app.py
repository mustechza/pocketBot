import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# ================= CONFIG =================
API_URL = "https://crashscores.com/api/games/rounds-detailed"
PARAMS = {
    "limit": 50,
    "brand": "sportpesa",
    "game": "spaceman"
}

# ================= STATE =================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_round_id" not in st.session_state:
    st.session_state.last_round_id = None

# ================= FETCH DATA =================
def fetch_data():
    try:
        res = requests.get(API_URL, params=PARAMS)
        data = res.json()
        return data.get("rounds", [])
    except:
        return []

# ================= SIGNAL ENGINE =================
def low_streak(data):
    count = 0
    for x in reversed(data):
        if x < 1.5:
            count += 1
        else:
            break
    return count

def volatility(data):
    if len(data) < 10:
        return 0
    return np.std(data[-10:])

def confidence_score(data):
    streak = low_streak(data)
    vol = volatility(data)

    score = 0
    if streak >= 5:
        score += 50
    if vol > 2:
        score += 50

    return score, streak, vol

# ================= UI =================
st.set_page_config(layout="wide")
st.title("🚀 Spaceman Crash Dashboard (Stable API)")

# ================= FETCH LOOP =================
rounds = fetch_data()

new_data = False

for r in reversed(rounds):
    rid = r["round_id"]
    coef = r["coefficient"]

    if st.session_state.last_round_id is None:
        st.session_state.last_round_id = rid

    if rid != st.session_state.last_round_id:
        st.session_state.history.append(coef)
        st.session_state.last_round_id = rid
        new_data = True

# limit history
if len(st.session_state.history) > 200:
    st.session_state.history = st.session_state.history[-200:]

# ================= LIVE DISPLAY =================
if rounds:
    latest = rounds[0]["coefficient"]
    st.metric("Latest Crash", f"{latest}x")
else:
    st.metric("Latest Crash", "Loading...")

# ================= ANALYSIS =================
history = st.session_state.history

if history:
    df = pd.DataFrame(history, columns=["Multiplier"])

    score, streak, vol = confidence_score(history)

    col1, col2, col3 = st.columns(3)
    col1.metric("Low Streak", streak)
    col2.metric("Volatility", round(vol, 2))
    col3.metric("Confidence", f"{score}%")

    # SIGNALS
    if score >= 70:
        st.success("🔥 STRONG SIGNAL - SPIKE ZONE")
    elif score >= 50:
        st.warning("⚠️ MODERATE SIGNAL")
    else:
        st.info("No strong signal")

    # CHART
    st.subheader("📈 Crash History")
    st.line_chart(df)

    # LAST ROUNDS
    st.subheader("📊 Last 20 Rounds")
    st.write(df.tail(20))

else:
    st.info("Waiting for data...")

# ================= REFRESH =================
time.sleep(3)
st.rerun()

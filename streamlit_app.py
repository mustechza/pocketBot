import streamlit as st
import socketio
import pandas as pd
import numpy as np
import threading
import time

# ================== GLOBAL STATE ==================
if "history" not in st.session_state:
    st.session_state.history = []

if "live_multiplier" not in st.session_state:
    st.session_state.live_multiplier = 0.0

if "connected" not in st.session_state:
    st.session_state.connected = False

# ================== SIGNAL ENGINE ==================
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

# ================== SOCKET CLIENT ==================
def run_socket():
    sio = socketio.Client()

    @sio.event
    def connect():
        st.session_state.connected = True
        print("Connected")

        # try multiple subscription formats
        sio.emit("subscribe", {"channel": "crash"})
        sio.emit("join", {"game": "crash"})

    @sio.event
    def disconnect():
        st.session_state.connected = False
        print("Disconnected")

    @sio.on("*")
    def catch_all(event, data):
        if isinstance(data, dict):

            # live tick
            if "multiplier" in data:
                st.session_state.live_multiplier = data["multiplier"]

            # crash end
            if "coefficient" in data or "multiplier" in data:
                crash = data.get("coefficient") or data.get("multiplier")

                if isinstance(crash, (int, float)):
                    st.session_state.history.append(crash)

                    # keep last 200
                    if len(st.session_state.history) > 200:
                        st.session_state.history = st.session_state.history[-200:]

    sio.connect(
        "https://socketv4.bc.game",
        transports=["websocket"]
    )

    sio.wait()

# ================== START SOCKET THREAD ==================
if "socket_thread_started" not in st.session_state:
    thread = threading.Thread(target=run_socket, daemon=True)
    thread.start()
    st.session_state.socket_thread_started = True

# ================== UI ==================
st.set_page_config(layout="wide")

st.title("🚀 BC.Game Crash Dashboard")

# STATUS
status = "🟢 Connected" if st.session_state.connected else "🔴 Disconnected"
st.subheader(f"Status: {status}")

# LIVE MULTIPLIER
st.metric("Live Multiplier", f"{st.session_state.live_multiplier}x")

# HISTORY DATAFRAME
history = st.session_state.history

if history:
    df = pd.DataFrame(history, columns=["Multiplier"])

    # SIGNALS
    score, streak, vol = confidence_score(history)

    col1, col2, col3 = st.columns(3)

    col1.metric("Low Streak", streak)
    col2.metric("Volatility", round(vol, 2))
    col3.metric("Confidence", f"{score}%")

    # SIGNAL ALERT
    if score >= 70:
        st.success("🔥 STRONG SIGNAL - SPIKE ZONE")
    elif score >= 50:
        st.warning("⚠️ MODERATE SIGNAL")

    # CHART
    st.subheader("📈 Crash History")
    st.line_chart(df)

    # LAST VALUES
    st.subheader("📊 Last 20 Rounds")
    st.write(df.tail(20))

else:
    st.info("Waiting for data...")

# AUTO REFRESH
time.sleep(1)
st.rerun()

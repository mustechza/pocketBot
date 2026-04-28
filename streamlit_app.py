import streamlit as st
import socketio
import pandas as pd
import numpy as np
import threading
import time


sio = socketio.Client(logger=True, engineio_logger=True)

@sio.event
def connect():
    print("✅ Connected")

    # Try multiple subscriptions
    sio.emit("subscribe", {"channel": "crash"})
    sio.emit("join", {"game": "crash"})
    sio.emit("sub", {"name": "crash"})

@sio.event
def disconnect():
    print("❌ Disconnected")

# 🔥 Catch EVERYTHING
@sio.on("*")
def catch_all(event, data):
    print("EVENT:", event)
    print("DATA:", data)
    print("-" * 50)

sio.connect(
    "https://socketv4.bc.game",
    transports=["websocket"]
)

sio.wait()
# ================== STATE INIT ==================
if "history" not in st.session_state:
    st.session_state.history = []

if "tick_history" not in st.session_state:
    st.session_state.tick_history = []

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

# ================== SOCKET ==================
def run_socket():
    sio = socketio.Client(reconnection=True)

    # -------- CONNECT --------
    @sio.event
    def connect():
        st.session_state.connected = True
        print("✅ Connected")

        sio.emit("subscribe", {"channel": "crash"})
        sio.emit("join", {"game": "crash"})

    @sio.event
    def disconnect():
        st.session_state.connected = False
        print("❌ Disconnected")

    # -------- LIVE TICKS --------
    @sio.on("crash_tick")
    def on_tick(data):
        try:
            multiplier = data.get("multiplier")

            if isinstance(multiplier, (int, float)) and 1 <= multiplier <= 1000:
                st.session_state.live_multiplier = multiplier

                st.session_state.tick_history.append(multiplier)

                # limit tick history
                if len(st.session_state.tick_history) > 50:
                    st.session_state.tick_history = st.session_state.tick_history[-50:]

        except Exception as e:
            print("Tick error:", e)

    # -------- CRASH END --------
    @sio.on("crash_end")
    def on_crash(data):
        try:
            crash = data.get("multiplier") or data.get("coefficient")

            if isinstance(crash, (int, float)) and 1 <= crash <= 1000:
                st.session_state.history.append(crash)

                # limit history
                if len(st.session_state.history) > 200:
                    st.session_state.history = st.session_state.history[-200:]

                # reset tick graph for next round
                st.session_state.tick_history = []

                print(f"💥 Crash: {crash}x")

        except Exception as e:
            print("Crash error:", e)

    # -------- FALLBACK --------
    @sio.on("round")
    def on_round(data):
        try:
            if "coefficient" in data:
                crash = data["coefficient"]

                if isinstance(crash, (int, float)):
                    st.session_state.history.append(crash)

                    if len(st.session_state.history) > 200:
                        st.session_state.history = st.session_state.history[-200:]

        except:
            pass

    # -------- CONNECT --------
    sio.connect(
        "https://socketv4.bc.game",
        transports=["websocket"]
    )

    sio.wait()

# ================== START THREAD ==================
if "socket_started" not in st.session_state:
    thread = threading.Thread(target=run_socket, daemon=True)
    thread.start()
    st.session_state.socket_started = True

# ================== UI ==================
st.set_page_config(layout="wide")
st.title("🚀 Crash Live Dashboard (BC.Game)")

# STATUS
status = "🟢 Connected" if st.session_state.connected else "🔴 Disconnected"
st.subheader(f"Status: {status}")

# LIVE MULTIPLIER
st.metric("Live Multiplier", f"{st.session_state.live_multiplier:.2f}x")

# LIVE GRAPH
st.subheader("📡 Live Multiplier (Current Round)")
if st.session_state.tick_history:
    st.line_chart(st.session_state.tick_history)
else:
    st.info("Waiting for round to start...")

# HISTORY + SIGNALS
history = st.session_state.history

if history:
    df = pd.DataFrame(history, columns=["Multiplier"])

    score, streak, vol = confidence_score(history)

    col1, col2, col3 = st.columns(3)
    col1.metric("Low Streak", streak)
    col2.metric("Volatility", round(vol, 2))
    col3.metric("Confidence", f"{score}%")

    # SIGNAL DISPLAY
    if score >= 70:
        st.success("🔥 STRONG SIGNAL - SPIKE ZONE")
    elif score >= 50:
        st.warning("⚠️ MODERATE SIGNAL")
    else:
        st.info("No strong signal")

    # HISTORY CHART
    st.subheader("📈 Crash History")
    st.line_chart(df)

    # LAST ROUNDS
    st.subheader("📊 Last 20 Rounds")
    st.write(df.tail(20))

else:
    st.info("Waiting for crash data...")

# AUTO REFRESH (smooth)
time.sleep(1)
st.rerun()

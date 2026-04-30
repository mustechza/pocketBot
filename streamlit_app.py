import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import threading
import time

# ================= STATE =================
if "live_multiplier" not in st.session_state:
    st.session_state.live_multiplier = 0.0

if "phase" not in st.session_state:
    st.session_state.phase = "Loading"

if "history" not in st.session_state:
    st.session_state.history = []

# ================= SELENIUM WORKER =================
def run_scraper():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=options)
    driver.get("https://crashscores.com")

    time.sleep(5)

    last_phase = None

    while True:
        try:
            stat_boxes = driver.find_elements(By.CSS_SELECTOR, ".live-stat-box")

            multiplier_el = stat_boxes[0].find_element(By.CLASS_NAME, "current-val")
            phase_el = stat_boxes[1].find_element(By.CLASS_NAME, "current-val")

            raw_value = multiplier_el.text.strip()
            phase = phase_el.text.strip()

            st.session_state.phase = phase

            if raw_value != "-x":
                value = raw_value.replace("x", "").strip()

                try:
                    multiplier = float(value)
                    st.session_state.live_multiplier = multiplier
                except:
                    pass

            # Detect round end
            if last_phase == "Running" and phase == "Idle":
                crash = st.session_state.live_multiplier
                if crash > 0:
                    st.session_state.history.append(crash)

                    # limit
                    if len(st.session_state.history) > 200:
                        st.session_state.history = st.session_state.history[-200:]

            last_phase = phase

        except Exception as e:
            print("Scraper error:", e)

        time.sleep(0.5)

# ================= START THREAD =================
if "scraper_started" not in st.session_state:
    thread = threading.Thread(target=run_scraper, daemon=True)
    thread.start()
    st.session_state.scraper_started = True

# ================= SIGNAL ENGINE =================
def low_streak(data):
    count = 0
    for x in reversed(data):
        if x < 1.5:
            count += 1
        else:
            break
    return count

# ================= UI =================
st.set_page_config(layout="wide")
st.title("🚀 Live Crash Dashboard (DOM Scraper)")

# STATUS
st.subheader(f"Phase: {st.session_state.phase}")

# LIVE MULTIPLIER
st.metric("Live Multiplier", f"{st.session_state.live_multiplier:.2f}x")

# HISTORY
history = st.session_state.history

if history:
    streak = low_streak(history)

    col1, col2 = st.columns(2)
    col1.metric("Low Streak", streak)
    col2.metric("Total Rounds", len(history))

    st.subheader("📈 Crash History")
    st.line_chart(history)

    st.subheader("📊 Last 20")
    st.write(history[-20:])

else:
    st.info("Waiting for data...")

# REFRESH LOOP
time.sleep(1)
st.rerun()

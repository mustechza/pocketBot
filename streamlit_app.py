import streamlit as st
import time
import threading
import queue
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from streamlit_autorefresh import st_autorefresh

# ================= QUEUE (SAFE COMMUNICATION) =================
data_queue = queue.Queue()

# ================= SCRAPER =================
def run_scraper(q: queue.Queue):
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

            multiplier = 0.0

            if raw_value != "-x":
                try:
                    multiplier = float(raw_value.replace("x", "").strip())
                except:
                    pass

            event = {
                "multiplier": multiplier,
                "phase": phase
            }

            q.put(event)

            last_phase = phase

        except Exception as e:
            print("Scraper error:", e)

        time.sleep(0.5)

# ================= START SCRAPER THREAD =================
if "started" not in st.session_state:
    threading.Thread(target=run_scraper, args=(data_queue,), daemon=True).start()
    st.session_state.started = True

# ================= SESSION STATE =================
if "live_multiplier" not in st.session_state:
    st.session_state.live_multiplier = 0.0

if "phase" not in st.session_state:
    st.session_state.phase = "Loading"

if "history" not in st.session_state:
    st.session_state.history = []

# ================= STREAM DATA FROM QUEUE =================
while not data_queue.empty():
    data = data_queue.get()

    st.session_state.live_multiplier = data["multiplier"]
    st.session_state.phase = data["phase"]

    if data["phase"] == "Idle" and data["multiplier"] > 0:
        st.session_state.history.append(data["multiplier"])

        if len(st.session_state.history) > 200:
            st.session_state.history = st.session_state.history[-200:]

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
st.title("🚀 Live Crash Dashboard (Stable Version)")

st.subheader(f"Phase: {st.session_state.phase}")
st.metric("Live Multiplier", f"{st.session_state.live_multiplier:.2f}x")

history = st.session_state.history

if history:
    col1, col2 = st.columns(2)
    col1.metric("Low Streak", low_streak(history))
    col2.metric("Total Rounds", len(history))

    st.subheader("📈 Crash History")
    st.line_chart(history)

    st.subheader("📊 Last 20")
    st.write(history[-20:])
else:
    st.info("Waiting for data...")

# ================= AUTO REFRESH =================
st_autorefresh(interval=1000, key="refresh")

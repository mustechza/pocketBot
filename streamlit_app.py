import streamlit as st
import threading
import time
import queue
from playwright.sync_api import sync_playwright
from streamlit_autorefresh import st_autorefresh

# ================= QUEUE =================
data_queue = queue.Queue()

# ================= SCRAPER =================
def scrape_bc_game(q: queue.Queue):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto("https://bc.game/game/crash", timeout=60000)
        page.wait_for_timeout(8000)

        last_seen = set()

        while True:
            try:
                rows = page.query_selector_all("tbody tr")

                for row in rows[:30]:  # latest 30
                    cols = row.query_selector_all("td")

                    if len(cols) < 2:
                        continue

                    game_id = cols[0].inner_text().strip()
                    multiplier_text = cols[1].inner_text().strip()

                    try:
                        multiplier = float(multiplier_text)
                    except:
                        continue

                    if game_id not in last_seen:
                        last_seen.add(game_id)

                        q.put({
                            "id": game_id,
                            "multiplier": multiplier
                        })

                # keep memory small
                if len(last_seen) > 500:
                    last_seen = set(list(last_seen)[-300:])

            except Exception as e:
                print("Scraper error:", e)

            time.sleep(2)

# ================= START THREAD =================
if "started" not in st.session_state:
    threading.Thread(target=scrape_bc_game, args=(data_queue,), daemon=True).start()
    st.session_state.started = True

# ================= STATE =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= LOAD QUEUE =================
while not data_queue.empty():
    data = data_queue.get()
    st.session_state.history.append(data["multiplier"])

    if len(st.session_state.history) > 200:
        st.session_state.history = st.session_state.history[-200:]

# ================= ANALYSIS =================
def low_streak(data):
    count = 0
    for x in reversed(data):
        if x < 1.5:
            count += 1
        else:
            break
    return count

def high_hits(data):
    return len([x for x in data[-50:] if x >= 10])

# ================= UI =================
st.set_page_config(layout="wide")
st.title("🚀 BC.Game Crash Live Tracker (Playwright)")

history = st.session_state.history

if history:
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rounds", len(history))
    col2.metric("Low Streak (<1.5x)", low_streak(history))
    col3.metric("High Hits (10x+ last 50)", high_hits(history))

    st.subheader("📈 Crash History")
    st.line_chart(history)

    st.subheader("📊 Last 20 Crashes")
    st.write(history[-20:])

else:
    st.info("Waiting for crash data...")

# auto refresh
st_autorefresh(interval=2000, key="refresh")

import streamlit as st
st.set_page_config(page_title="Pocket Option Monitor", layout="wide")  # MUST be first Streamlit call

from streamlit_autorefresh import st_autorefresh
import pandas as pd
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import plotly.express as px

# --- AUTO REFRESH ---
st_autorefresh(interval=10 * 1000, key="datarefresh")  # every 10 seconds

# --- TITLE ---
st.title("📊 Pocket Option Market Monitor (Real-Time)")

# --- GET SELENIUM DRIVER ---
def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    return driver

# --- SCRAPE MARKET DATA ---
def scrape_market_data():
    driver = get_driver()
    driver.get("https://pocketoption.com/en/")

    time.sleep(5)  # wait for data to load

    try:
        table = driver.find_element(By.CLASS_NAME, "asset-table__body")
        rows = table.find_elements(By.CLASS_NAME, "asset-table__row")

        data = []
        for row in rows:
            try:
                asset = row.find_element(By.CLASS_NAME, "asset-table__title").text.strip()
                payout = row.find_element(By.CLASS_NAME, "asset-table__profit").text.strip().replace("%", "")
                data.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Asset": asset,
                    "Payout (%)": float(payout)
                })
            except Exception:
                continue

        df = pd.DataFrame(data)
        driver.quit()
        return df
    except Exception as e:
        driver.quit()
        st.error(f"Error scraping data: {e}")
        return pd.DataFrame()

# --- FETCH + DISPLAY DATA ---
df = scrape_market_data()

if not df.empty:
    st.dataframe(df, use_container_width=True)

    # --- CHART ---
    fig = px.bar(df, x="Asset", y="Payout (%)", color="Payout (%)",
                 title="Current Payouts by Asset", labels={"Payout (%)": "Payout %"})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available. Please try again later.")

import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# --- Setup Headless Chrome ---
@st.cache_resource
def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    return driver

# --- Scrape Market Data from Pocket Option ---
def scrape_market_data(driver):
    driver.get("https://pocketoption.com/en/")
    time.sleep(5)  # Wait for data to load

    market_data = []
    try:
        rows = driver.find_elements(By.CSS_SELECTOR, ".asset-list__item")
        for row in rows:
            name = row.find_element(By.CSS_SELECTOR, ".asset__name").text
            payout = row.find_element(By.CSS_SELECTOR, ".asset__payout").text
            market_data.append({"Asset": name, "Payout": payout})
    except Exception as e:
        st.error(f"Error scraping data: {e}")
        return pd.DataFrame()

    return pd.DataFrame(market_data)

# --- Streamlit UI ---
st.set_page_config(page_title="Pocket Option Market Monitor", layout="wide")
st.title("📊 Pocket Option Market Monitor")

driver = get_driver()

if st.button("Refresh Market Data"):
    df = scrape_market_data(driver)
    if not df.empty:
        st.dataframe(df)
    else:
        st.warning("No data found.")

st.caption("Data is scraped live from Pocket Option.")

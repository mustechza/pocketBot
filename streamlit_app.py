import streamlit as st
import pandas as pd
import asyncio
from playwright.async_api import async_playwright

# Set up Streamlit page config
st.set_page_config(page_title="Pocket Option Monitor", layout="wide")

# Function to scrape market data using Playwright
async def scrape_market_data():
    async with async_playwright() as p:
        # Launch Chromium in headless mode
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Navigate to Pocket Option market page
        await page.goto("https://pocketoption.com")

        # Wait for page load and the specific market data element (adjust selector as needed)
        await page.wait_for_selector(".market-table")  # Adjust this selector

        # Scrape the required data
        rows = await page.query_selector_all(".market-table tr")  # Adjust based on actual table row

        # Parse market data into a list of dictionaries
        market_data = []
        for row in rows:
            cells = await row.query_selector_all("td")
            market_data.append({
                "Market": await cells[0].inner_text(),
                "Price": await cells[1].inner_text(),
                "Change": await cells[2].inner_text(),
            })

        # Close the browser
        await browser.close()

        return pd.DataFrame(market_data)

# Function to refresh and display data
def display_market_data():
    # Scrape market data
    data = asyncio.run(scrape_market_data())
    
    # Display in Streamlit
    st.write("## Real-time Pocket Option Market Data")
    st.dataframe(data)

    # Add chart if applicable
    if not data.empty:
        st.line_chart(data["Price"].astype(float))

# Streamlit layout
st.title("Pocket Option Real-Time Market Monitor")

# Auto refresh the data every 60 seconds
st.cache_data(ttl=60, show_spinner=True)(display_market_data)

# Display market data on the Streamlit page
display_market_data()


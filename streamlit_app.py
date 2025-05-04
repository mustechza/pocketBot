import streamlit as st
from binance import Client
from binance.exceptions import BinanceAPIException
import pandas as pd

st.set_page_config(page_title="Binance Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .stDataFrame, .stMetric, .stTextInput {
            background-color: #0e1117;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Binance Market Dashboard")

# Load API keys from Streamlit secrets
try:
    API_KEY = st.secrets["binance"]["API_KEY"]
    API_SECRET = st.secrets["binance"]["API_SECRET"]
    client = Client(API_KEY, API_SECRET)

    # Check connectivity
    client.ping()
    st.sidebar.success("✅ Connected to Binance with API credentials.")
    use_private = True
except Exception as e:
    st.sidebar.warning("⚠️ Using public access only (no API credentials or invalid key).")
    client = Client()  # fallback with no keys
    use_private = False

# --- Market Depth ---
st.subheader("Order Book (Market Depth)")
symbol = st.sidebar.text_input("Symbol (e.g., BTCUSDT)", value="BTCUSDT").upper()
depth = {}

try:
    depth = client.get_order_book(symbol=symbol)
    bids = pd.DataFrame(depth['bids'], columns=['Price', 'Quantity']).astype(float)
    asks = pd.DataFrame(depth['asks'], columns=['Price', 'Quantity']).astype(float)

    col1, col2 = st.columns(2)
    with col1:
        st.write("💰 Top 5 Bids")
        st.dataframe(bids.head(), use_container_width=True)
    with col2:
        st.write("💸 Top 5 Asks")
        st.dataframe(asks.head(), use_container_width=True)
except Exception as e:
    st.error(f"Error fetching order book for {symbol}: {e}")

# --- Price Ticker ---
st.subheader("Live Symbol Prices")
try:
    prices = client.get_all_tickers()
    df_prices = pd.DataFrame(prices)
    st.dataframe(df_prices.head(20), use_container_width=True)
except Exception as e:
    st.error(f"Error fetching ticker prices: {e}")

# --- Historical Klines ---
st.subheader("Historical Kline Data")
interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=0)
try:
    klines = client.get_historical_klines(symbol, interval, "1 day ago UTC")
    df_klines = pd.DataFrame(klines, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Vol", "Taker Buy Quote Vol", "Ignore"
    ])
    df_klines["Open Time"] = pd.to_datetime(df_klines["Open Time"], unit='ms')
    df_klines = df_klines[["Open Time", "Open", "High", "Low", "Close", "Volume"]].astype({
        "Open": float, "High": float, "Low": float, "Close": float, "Volume": float
    })
    st.line_chart(df_klines.set_index("Open Time")[["Open", "Close"]])
except Exception as e:
    st.error(f"Error fetching klines for {symbol}: {e}")

# --- Withdrawals (Private Only) ---
if use_private:
    st.subheader("Withdrawal History (ETH)")
    try:
        eth_withdraws = client.get_withdraw_history(coin="ETH")
        df_wd = pd.DataFrame(eth_withdraws)
        if not df_wd.empty:
            st.dataframe(df_wd[["amount", "address", "status", "applyTime"]].head(), use_container_width=True)
        else:
            st.info("No ETH withdrawals found.")
    except BinanceAPIException as e:
        st.warning(f"Could not fetch withdrawals: {e.message}")
else:
    st.info("🔒 API key required to view withdrawal history.")


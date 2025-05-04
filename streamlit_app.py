from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from binance.exceptions import BinanceAPIException
import os

# Set your API credentials securely (avoid hardcoding!)
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

# === 1. Market Depth ===
depth = client.get_order_book(symbol='BNBBTC')
print(depth)

# === 2. Test Buy Order ===
test_order = client.create_test_order(
    symbol='BNBBTC',
    side=Client.SIDE_BUY,
    type=Client.ORDER_TYPE_MARKET,
    quantity=100
)

# === 3. All Symbol Prices ===
prices = client.get_all_tickers()
print(prices)

# === 4. Withdrawals (REAL money, use with caution) ===
try:
    result = client.withdraw(
        coin='ETH',
        address='<eth_address>',  # replace with actual ETH address
        amount=100
    )
    print("Withdraw Success:", result)
except BinanceAPIException as e:
    print("Withdraw Error:", e)

# === 5. Withdrawal History ===
withdrawals = client.get_withdraw_history()
eth_withdrawals = client.get_withdraw_history(coin='ETH')
print(withdrawals)
print(eth_withdrawals)

# === 6. Get Deposit Address ===
btc_address = client.get_deposit_address(coin='BTC')
print("BTC Deposit Address:", btc_address)

# === 7. Historical Klines ===
klines_day = client.get_historical_klines("BNBBTC", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
klines_month = client.get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")
klines_week = client.get_historical_klines("NEOBTC", Client.KLINE_INTERVAL_1WEEK, "1 Jan, 2017")

# === 8. Websocket Managers ===
def handle_socket_message(msg):
    print(f"Websocket message type: {msg['e']}")
    print(msg)

def handle_dcm_message(depth_cache):
    print(f"Symbol: {depth_cache.symbol}")
    print("Top 5 Bids:", depth_cache.get_bids()[:5])
    print("Top 5 Asks:", depth_cache.get_asks()[:5])
    print("Last Update:", depth_cache.update_time)

# Start Websocket
twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
twm.start()
twm.start_kline_socket(callback=handle_socket_message, symbol='BNBBTC')

# Start Depth Cache
dcm = ThreadedDepthCacheManager()
dcm.start()
dcm.start_depth_cache(callback=handle_dcm_message, symbol='ETHBTC')

# Optional: Options Symbol Depth Cache
options_symbol = 'BTC-210430-36000-C'  # make sure it's valid
dcm.start_options_depth_cache(callback=handle_dcm_message, symbol=options_symbol)

import requests
import time

BASE_URL = 'https://api.binance.com/api/v3'

def get_price(symbol):
    """Get current price for a trading pair"""
    endpoint = f'/ticker/price'
    params = {'symbol': symbol}
    return _make_request(endpoint, params)

def get_klines(symbol, interval, **kwargs):
    """Get historical klines/candlestick data
    Args:
        symbol: trading pair (e.g., BTCUSDT)
        interval: timeframe interval (e.g., 1m, 5m, 1h, 1d)
        kwargs: optional parameters (limit, startTime, endTime)
    """
    endpoint = f'/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        **kwargs
    }
    return _make_request(endpoint, params)

def get_orderbook(symbol, limit=100):
    """Get order book data
    Args:
        symbol: trading pair
        limit: number of orders to return (default 100, max 5000)
    """
    endpoint = f'/depth'
    params = {'symbol': symbol, 'limit': limit}
    return _make_request(endpoint, params)

def _make_request(endpoint, params=None):
    """Handle API requests"""
    try:
        response = requests.get(
            BASE_URL + endpoint,
            params=params,
            timeout=5  # Timeout in seconds
        )
        response.raise_for_status()  # Raise exception for bad status codes
        
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except ValueError as e:
        print(f"Failed to parse JSON response: {e}")
        return None

# Example usage
if __name__ == '__main__':
    # Get current BTC price
    btc_price = get_price('BTCUSDT')
    print("Current BTC/USDT Price:", btc_price)

    # Get ETH/BTC historical data (last 5 1-hour candles)
    eth_klines = get_klines('ETHBTC', '1h', limit=5)
    if eth_klines:
        print("\nETH/BTC 1-hour Candles:")
        for kline in eth_klines:
            # Convert timestamp to readable format
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', 
                                    time.localtime(kline[0]/1000))
            print(f"{timestamp} - Open: {kline[1]} High: {kline[2]} "
                f"Low: {kline[3]} Close: {kline[4]}")

    # Get BTCUSDT order book
    order_book = get_orderbook('BTCUSDT', limit=10)
    if order_book:
        print("\nBTC/USDT Order Book (Top 10):")
        print("Bids:")
        for bid in order_book['bids']:
            print(f"Price: {bid[0]} Quantity: {bid[1]}")
        print("\nAsks:")
        for ask in order_book['asks']:
            print(f"Price: {ask[0]} Quantity: {ask[1]}")

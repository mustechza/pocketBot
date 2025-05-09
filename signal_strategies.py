
import pandas_ta as ta

def ema_crossover_signal(df):
    df['ema_fast'] = ta.ema(df['close'], length=5)
    df['ema_slow'] = ta.ema(df['close'], length=13)

    if df['ema_fast'].iloc[-2] < df['ema_slow'].iloc[-2] and df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1]:
        return "BUY"
    elif df['ema_fast'].iloc[-2] > df['ema_slow'].iloc[-2] and df['ema_fast'].iloc[-1] < df['ema_slow'].iloc[-1]:
        return "SELL"
    return None

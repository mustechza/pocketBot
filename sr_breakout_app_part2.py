
def detect_signals(df, strategy):
    signals = []
    for i in range(1, len(df)):
        t = df['timestamp'].iloc[i]
        price = df['close'].iloc[i]

        if strategy == "EMA Cross":
            if df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i]:
                signals.append(generate_signal(t, "Buy (EMA Cross)", price))
            elif df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i]:
                signals.append(generate_signal(t, "Sell (EMA Cross)", price))

        elif strategy == "RSI Divergence":
            rsi = df['RSI'].iloc[i]
            if rsi < 30:
                signals.append(generate_signal(t, "Buy (RSI Oversold)", price))
            elif rsi > 70:
                signals.append(generate_signal(t, "Sell (RSI Overbought)", price))

        elif strategy == "MACD Cross":
            if df['MACD'].iloc[i-1] < df['MACD_signal'].iloc[i-1] and df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                signals.append(generate_signal(t, "Buy (MACD Cross)", price))
            elif df['MACD'].iloc[i-1] > df['MACD_signal'].iloc[i-1] and df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
                signals.append(generate_signal(t, "Sell (MACD Cross)", price))

        elif strategy == "Bollinger Band Bounce":
            if df['close'].iloc[i] < df['BB_lower'].iloc[i]:
                signals.append(generate_signal(t, "Buy (BB Lower)", price))
            elif df['close'].iloc[i] > df['BB_upper'].iloc[i]:
                signals.append(generate_signal(t, "Sell (BB Upper)", price))

        elif strategy == "Stochastic Oscillator":
            stoch = df['Stochastic'].iloc[i]
            if stoch < 20:
                signals.append(generate_signal(t, "Buy (Stochastic)", price))
            elif stoch > 80:
                signals.append(generate_signal(t, "Sell (Stochastic)", price))

        elif strategy == "EMA + RSI Combined":
            if df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i] and df['RSI'].iloc[i] < 40:
                signals.append(generate_signal(t, "Buy (EMA+RSI)", price))
            elif df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i] and df['RSI'].iloc[i] > 60:
                signals.append(generate_signal(t, "Sell (EMA+RSI)", price))

    return signals

def train_ml_model(df):
    df = df.copy()
    df['target'] = (df['close'].shift(-2) > df['close']).astype(int)
    features = ['EMA5', 'EMA20', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Stochastic']
    df = df.dropna(subset=features + ['target'])

    X = df[features]
    y = df['target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    df['ML_Prediction'] = model.predict(X)
    df['Signal'] = df['ML_Prediction'].map({1: 'Buy (ML)', 0: 'Sell (ML)'})
    df['Trade Duration (min)'] = 2

    return df[['timestamp', 'Signal', 'close', 'Trade Duration (min)']].rename(columns={'close': 'Price'})

def simulate_money_management(signals, strategy="Flat", initial_balance=1000, bet_size=10):
    balance = initial_balance
    last_bet = bet_size
    wins, losses, pnl = 0, 0, []

    result_log = []
    for s in signals:
        win = np.random.choice([True, False], p=[0.55, 0.45])
        if win:
            balance += last_bet
            result = "Win"
            wins += 1
            last_bet = bet_size
        else:
            balance -= last_bet
            result = "Loss"
            losses += 1
            if strategy == "Martingale":
                last_bet *= 2
        pnl.append(balance)
        result_log.append({
            "Time": s["Time"], "Signal": s["Signal"], "Result": result,
            "Balance": balance, "Trade Duration (min)": s["Trade Duration (min)"]
        })

    df = pd.DataFrame(result_log)
    max_drawdown = ((df['Balance'].cummax() - df['Balance']) / df['Balance'].cummax()).max()
    roi = ((balance - initial_balance) / initial_balance) * 100
    profit_factor = (wins * bet_size) / max(losses * bet_size, 1)

    return df, {
        "Win Rate (%)": 100 * wins / (wins + losses),
        "ROI (%)": roi,
        "Max Drawdown (%)": 100 * max_drawdown,
        "Profit Factor": profit_factor
    }

def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA5'], name="EMA5", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA20'], name="EMA20", line=dict(color='red')))
    fig.update_layout(title=asset, xaxis_rangeslider_visible=False)
    return fig

def show_browser_alert(message):
    js_code = f"""
    <script>
    alert("{message}");
    </script>
    """
    components.html(js_code)

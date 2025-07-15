import ta
import pandas as pd
df = pd.read_csv('aapl_data.csv', index_col=0, parse_dates=True)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Make a copy of your original DataFrame
df_ta = df.copy()

# Relative Strength Index (RSI)
df_ta['RSI'] = ta.momentum.RSIIndicator(close=df_ta['Close'], window=14).rsi()

# Exponential Moving Average (EMA)
df_ta['EMA_20'] = ta.trend.EMAIndicator(close=df_ta['Close'], window=20).ema_indicator()

# Moving Average Convergence Divergence (MACD)
macd = ta.trend.MACD(close=df_ta['Close'])
df_ta['MACD'] = macd.macd()
df_ta['MACD_signal'] = macd.macd_signal()

# Bollinger Bands
bb = ta.volatility.BollingerBands(close=df_ta['Close'], window=20)
df_ta['BB_high'] = bb.bollinger_hband()
df_ta['BB_low'] = bb.bollinger_lband()

# Drop NaNs from initial calculations
df_ta.dropna(inplace=True)

# View updated DataFrame
df_ta.tail()

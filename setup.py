import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Download historical data
ticker = "AAPL"
start_date = "2015-01-01"
end_date = "2024-12-31"

df = yf.download(ticker, start=start_date, end=end_date)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)  # Just in case
df.head()

# Plot the data
plt.figure(figsize=(14, 6))
plt.plot(df['Close'], label='AAPL Stock Price')
plt.title(f'{ticker} Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
scaled_df.head()

print(scaled_df.head())

df.to_csv('aapl_data.csv')
scaled_df.to_csv('aapl_scaled.csv')

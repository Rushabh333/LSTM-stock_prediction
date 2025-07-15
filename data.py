import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


window_size = 60 

scaled_df = pd.read_csv('aapl_scaled.csv', index_col=0)
data = scaled_df.values

X = []
y = []

for i in range(window_size, len(data)):
    X.append(data[i-window_size:i, :])
    y.append(data[i, 0])

X, y = np.array(X), np.array(y)

print("Input shape (X):", X.shape)  # (samples, time_steps, features)
print("Target shape (y):", y.shape)  # (samples,)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))



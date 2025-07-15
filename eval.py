import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from model import DeepLSTMModel  # Import the model class
import json
import os

# Load scaler and data
from sklearn.preprocessing import MinMaxScaler

# Load the scaler and data used for training
scaled_df = pd.read_csv('aapl_scaled.csv', index_col=0)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    scaled_df[col] = pd.to_numeric(scaled_df[col], errors='coerce')
scaled_df.dropna(inplace=True)
scaler = MinMaxScaler()
raw_df = pd.read_csv('aapl_data.csv', index_col=0)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
raw_df.dropna(inplace=True)
numeric_df = raw_df
scaler.fit(numeric_df)

# Prepare validation data (same as in hyper_param.py/model.py)
window_size = 60
data = scaled_df.values
X = []
y = []
for i in range(window_size, len(data)):
    X.append(data[i-window_size:i, :])
    y.append(data[i, 0])
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Use the same split as before
test_size = 0.2
split_idx = int(len(X) * (1 - test_size))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Device agnostic
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Try to load best hyperparameters from file
best_hyperparams_path = 'best_hyperparams.json'
if os.path.exists(best_hyperparams_path):
    with open(best_hyperparams_path, 'r') as f:
        best_hyperparams = json.load(f)
    best_units = best_hyperparams['units']
    best_dropout = best_hyperparams['dropout']
    input_size = best_hyperparams.get('input_size', X.shape[2])
else:
    best_units = 64  # fallback value
    best_dropout = 0.32579501531686833  # fallback value
    input_size = X.shape[2]

# Load the trained model
model = DeepLSTMModel(input_size=input_size, hidden_size=best_units, dropout=best_dropout)
model.load_state_dict(torch.load("best_lstm_model.pth", map_location=device))
model.to(device)
model.eval()

predictions = []
targets = []

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        output = model(xb).cpu().numpy()
        predictions.extend(output.flatten())
        targets.extend(yb.numpy().flatten())

plt.figure(figsize=(14, 6))
plt.plot(targets, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='orange')
plt.title('LSTM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Scaled Close Price')
plt.legend()
plt.grid(True)
plt.show()

rmse = np.sqrt(mean_squared_error(targets, predictions))
print("Validation RMSE:", rmse)

# Unscale predictions and actuals for 'Close' column
# This assumes 'Close' is the last column in the original data
preds_unscaled = scaler.inverse_transform(
    np.hstack([
        np.zeros((len(predictions), numeric_df.shape[1] - 1)),
        np.array(predictions).reshape(-1, 1)
    ])
)[:, -1]

actual_unscaled = scaler.inverse_transform(
    np.hstack([
        np.zeros((len(targets), numeric_df.shape[1] - 1)),
        np.array(targets).reshape(-1, 1)
    ])
)[:, -1]

plt.figure(figsize=(14, 6))
plt.plot(actual_unscaled, label='Actual Unscaled', color='blue')
plt.plot(preds_unscaled, label='Predicted Unscaled', color='orange')
plt.title('LSTM Stock Price Prediction (Unscaled)')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

rmse_unscaled = np.sqrt(mean_squared_error(actual_unscaled, preds_unscaled))
print("Validation RMSE (unscaled):", rmse_unscaled)

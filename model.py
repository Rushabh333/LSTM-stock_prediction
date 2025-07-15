import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# Load and process the data
scaled_df = pd.read_csv('aapl_scaled.csv', index_col=0)

# Convert all price/volume columns to numeric
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    scaled_df[col] = pd.to_numeric(scaled_df[col], errors='coerce')

# Drop rows with NaN (if any conversion fails)
scaled_df.dropna(inplace=True)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(scaled_df)  # scaled_df should now have only float columns
scaled_df = pd.DataFrame(scaled_data, index=scaled_df.index, columns=scaled_df.columns)


window_size = 60
X = []
y = []
data = scaled_df.values
for i in range(window_size, len(data)):
    X.append(data[i-window_size:i, :])
    y.append(data[i, 0])
# Convert to float32 arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Create datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

class DeepLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(DeepLSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                             num_layers=1, batch_first=True)
        
        self.lstm3 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                             num_layers=1, batch_first=True)
        
        self.lstm4 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                             num_layers=1, batch_first=True)

        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)    # final LSTM layer (no return_sequences)

        out = self.dropout2(out[:, -1, :])  # take last timestep output
        out = self.fc(out)
        return out

# Example usage (replace with your hyperparameter search or training loop)
hidden_units = 64  # example value
input_size = X.shape[2]
dropout_rate = 0.2  # example value
model = DeepLSTMModel(input_size=input_size, hidden_size=hidden_units, dropout=dropout_rate)

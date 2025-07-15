import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

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

data = scaled_df.values
window_size = 60
X = []
y = []
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

space = {
    "units": hp.choice("units", [32, 64, 128]),
    "dropout": hp.uniform("dropout", 0.1, 0.5),
    "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-2)),
    "batch_size": hp.choice("batch_size", [32, 64, 128])
}

# Device agnostic code for Mac (MPS), CUDA, or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def objective(space):
    # Data prep
    batch_size = space["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = DeepLSTMModel(
        input_size=X.shape[2],
        hidden_size=int(space["units"]),
        dropout=space["dropout"]
    )
    model.to(device)

    # Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=space["lr"])

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            rmse = torch.sqrt(loss)
            rmse.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    val_rmses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            mse = criterion(pred, yb)
            rmse = torch.sqrt(mse)
            val_rmses.append(rmse.item())

    avg_val_rmse = np.mean(val_rmses)
    print(f"VAL_RMSE: {avg_val_rmse:.6f} | space: {space}")
    
    return {"loss": avg_val_rmse, "status": STATUS_OK}

trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=25,
    trials=trials
)

units_options = [32, 64, 128]
batch_size_options = [32, 64, 128]

actual_best = best.copy()
actual_best['units'] = units_options[best['units']]
actual_best['batch_size'] = batch_size_options[best['batch_size']]

print("\nBest Hyperparameters (actual values):")
print(actual_best)

# --- Retrain the Best Model ---

# Use the actual best hyperparameters
best_units = actual_best['units']
best_dropout = actual_best['dropout']
best_lr = actual_best['lr']
best_batch_size = actual_best['batch_size']

# Optionally, combine train and val for final training
X_final = np.concatenate([X_train, X_val], axis=0)
y_final = np.concatenate([y_train, y_val], axis=0)

X_final_tensor = torch.tensor(X_final, dtype=torch.float32)
y_final_tensor = torch.tensor(y_final, dtype=torch.float32).unsqueeze(1)
final_dataset = TensorDataset(X_final_tensor, y_final_tensor)
final_loader = DataLoader(final_dataset, batch_size=best_batch_size, shuffle=False)

final_model = DeepLSTMModel(input_size=X.shape[2], hidden_size=best_units, dropout=best_dropout)
final_model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=best_lr)

epochs = 20  # You can increase this for final training
for epoch in range(epochs):
    final_model.train()
    epoch_losses = []
    for xb, yb in final_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        output = final_model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(epoch_losses):.6f}")

# Save the trained model
torch.save(final_model.state_dict(), "best_lstm_model.pth")
print("Best model retrained and saved as best_lstm_model.pth")

# Save best hyperparameters for evaluation
import json
with open('best_hyperparams.json', 'w') as f:
    json.dump({
        'units': best_units,
        'dropout': best_dropout,
        'input_size': X.shape[2]
    }, f)
print("Best hyperparameters saved to best_hyperparams.json")

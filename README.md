# LSTM Stock Price Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project implements an end-to-end pipeline for stock price prediction using a deep LSTM model. It includes data download, preprocessing, feature scaling, hyperparameter optimization with Hyperopt, model training, and evaluation. The code is device-agnostic and supports Mac (MPS), CUDA, and CPU.

## Features
- Downloads historical stock data using yfinance
- Preprocesses and scales data for LSTM
- Hyperparameter optimization with Hyperopt
- Deep LSTM model implemented in PyTorch
- Device-agnostic (MPS, CUDA, CPU)
- Evaluation and visualization of predictions

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Rushabh333/LSTM-stock_prediction.git
cd LSTM
```

### 2. Install dependencies
It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
If you don't have a `requirements.txt`, install manually:
```bash
pip install yfinance pandas matplotlib scikit-learn torch hyperopt
```

### 3. Download and preprocess data
```bash
python setup.py
```
This will download the data, scale it, and save `aapl_data.csv` and `aapl_scaled.csv`.

### 4. Hyperparameter search and model training
```bash
python hyper_param.py
```
This will run hyperparameter optimization, retrain the best model, and save `best_lstm_model.pth` and `best_hyperparams.json`.

### 5. Evaluate the model
```bash
python eval.py
```
This will plot predictions and print RMSE metrics.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 

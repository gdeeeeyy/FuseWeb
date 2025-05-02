# AI for Financial Data Generation – Use AI to generate synthetic financial market trends and analyze potential trading strategies.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# --- Generate Sample Real Data (Simulated Stock Prices) ---
def generate_real_data(n_series=1, length=1000):
    x = np.linspace(0, 50, length)
    prices = []
    for _ in range(n_series):
        noise = np.random.normal(0, 0.2, size=length)
        trend = np.sin(x) + 0.1 * x + noise
        prices.append(trend)
    return np.array(prices).reshape(n_series, length, 1)

real_data = generate_real_data(n_series=100)

# --- Build LSTM Autoencoder as a Simple Generative Model ---
timesteps = real_data.shape[1]
features = real_data.shape[2]

model = Sequential([
    LSTM(128, activation='relu', input_shape=(timesteps, features), return_sequences=False),
    RepeatVector(timesteps),
    LSTM(128, activation='relu', return_sequences=True),
    TimeDistributed(Dense(1))
])
model.compile(optimizer='adam', loss='mse')
model.fit(real_data, real_data, epochs=50, batch_size=16, verbose=0)

# --- Generate Synthetic Data ---
def generate_synthetic_samples(model, n_samples=10):
    seed = generate_real_data(n_series=n_samples)
    return model.predict(seed)

synthetic_data = generate_synthetic_samples(model, n_samples=5)

# --- Plot Real vs Synthetic Data ---
plt.figure(figsize=(12, 5))
plt.plot(real_data[0], label="Real")
plt.plot(synthetic_data[0], label="Synthetic", linestyle='--')
plt.legend()
plt.title("Real vs. Synthetic Financial Time Series")
plt.show()

# --- Strategy Backtest: Moving Average Crossover ---
def moving_average_strategy(prices, short_window=10, long_window=50):
    df = pd.DataFrame(prices, columns=["Price"])
    df["Short_MA"] = df["Price"].rolling(window=short_window).mean()
    df["Long_MA"] = df["Price"].rolling(window=long_window).mean()
    df["Signal"] = 0
    df["Signal"][short_window:] = np.where(
        df["Short_MA"][short_window:] > df["Long_MA"][short_window:], 1, 0
    )
    df["Position"] = df["Signal"].diff()
    return df

synthetic_series = synthetic_data[0].flatten()
strategy_df = moving_average_strategy(synthetic_series)

# --- Plot Trading Signals ---
plt.figure(figsize=(12, 6))
plt.plot(synthetic_series, label='Synthetic Price')
plt.plot(strategy_df['Short_MA'], label='Short MA')
plt.plot(strategy_df['Long_MA'], label='Long MA')
buy_signals = strategy_df[strategy_df['Position'] == 1].index
sell_signals = strategy_df[strategy_df['Position'] == -1].index
plt.scatter(buy_signals, synthetic_series[buy_signals], marker='^', color='g', label='Buy Signal')
plt.scatter(sell_signals, synthetic_series[sell_signals], marker='v', color='r', label='Sell Signal')
plt.title("Strategy Backtest on Synthetic Data")
plt.legend()
plt.show()

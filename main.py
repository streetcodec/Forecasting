import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Close']]

# LMSA (Linear Model with Seasonal Adjustment)
def lmsa_forecast(data, period, n_future_days):
    decomposition = seasonal_decompose(data['Close'], model='additive', period=period)
    trend = decomposition.trend.fillna(method='bfill').fillna(method='ffill')
    seasonal = decomposition.seasonal[-period:]  # Last seasonal cycle
    x = np.arange(len(trend)).reshape(-1, 1)
    model = LinearRegression().fit(x, trend)
    future_x = np.arange(len(trend), len(trend) + n_future_days).reshape(-1, 1)
    trend_forecast = model.predict(future_x)
    seasonal_forecast = np.tile(seasonal, int(np.ceil(n_future_days / period)))[:n_future_days]
    return trend_forecast + seasonal_forecast

# ARIMA Forecast
def arima_forecast(data, order, n_future_days):
    model = ARIMA(data['Close'], order=order)
    model_fit = model.fit()
    return model_fit.forecast(steps=n_future_days)

# SARIMA Forecast
def sarima_forecast(data, order, seasonal_order, n_future_days):
    model = SARIMAX(data['Close'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit.forecast(steps=n_future_days)

# Exponential Smoothing Forecast
def exp_smoothing_forecast(data, n_future_days):
    model = ExponentialSmoothing(data['Close'], seasonal='additive', seasonal_periods=12)
    model_fit = model.fit()
    return model_fit.forecast(steps=n_future_days)

# MLP Forecast
def mlp_forecast(data, n_future_days):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(5, len(scaled_data)):
        X.append(scaled_data[i-5:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500).fit(X, y)
    future = scaled_data[-5:].flatten()
    predictions = []
    for _ in range(n_future_days):
        pred = model.predict(future.reshape(1, -1))
        predictions.append(pred[0])
        future = np.roll(future, -1)
        future[-1] = pred[0]
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# RNN (LSTM) Forecast
def rnn_forecast(data, n_future_days):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(5, len(scaled_data)):
        X.append(scaled_data[i-5:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)
    future = scaled_data[-5:].reshape((1, 5, 1))
    predictions = []
    for _ in range(n_future_days):
        pred = model.predict(future, verbose=0)
        predictions.append(pred[0, 0])
        future = np.roll(future, -1, axis=1)
        future[0, -1, 0] = pred[0, 0]
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def predict_and_plot_with_future(ticker, start_date, end_date, n_future_days=30):
    data = fetch_stock_data(ticker, start_date, end_date)
    future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=n_future_days)

    # Getting the predictions
    period = 30
    lmsa_preds = lmsa_forecast(data, period, n_future_days)
    arima_preds = arima_forecast(data, (5, 1, 0), n_future_days)
    sarima_preds = sarima_forecast(data, (1, 1, 1), (1, 1, 1, period), n_future_days)
    exp_preds = exp_smoothing_forecast(data, n_future_days)
    mlp_preds = mlp_forecast(data, n_future_days)
    rnn_preds = rnn_forecast(data, n_future_days)

    # Plot everything 
    plt.figure(figsize=(14, 8))
    plt.plot(data['Close'], label='Historical Data', color='blue')
    plt.plot(future_dates, lmsa_preds, label='LMSA', linestyle='--')
    plt.plot(future_dates, arima_preds, label='ARIMA', linestyle='--')
    plt.plot(future_dates, sarima_preds, label='SARIMA', linestyle='--')
    plt.plot(future_dates, exp_preds, label='Exponential Smoothing', linestyle='--')
    plt.plot(future_dates, mlp_preds, label='MLP', linestyle='--')
    plt.plot(future_dates, rnn_preds, label='RNN', linestyle='--')

    plt.title(f"Forecast Comparison for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()
    plt.show()

    return {
        "LMSA": lmsa_preds,
        "ARIMA": arima_preds,
        "SARIMA": sarima_preds,
        "Exponential Smoothing": exp_preds,
        "MLP": mlp_preds,
        "RNN": rnn_preds
    }

if __name__ == "__main__":
    ticker_symbol = "IBM"
    start = "2024-08-30"
    end = "2024-11-24"
    n_days = 30

    predictions = predict_and_plot_with_future(ticker_symbol, start, end, n_days)
    for model, preds in predictions.items():
        print(f"{model} Predictions: {preds}")

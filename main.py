import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from pandas_datareader import data as pdr
import pandas_ta as ta
from dataclasses import dataclass
import yfinance as yf
import matplotlib.pyplot as plt

def main():

    # Ensure reproducibility 
    seed_value = 1
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    @dataclass
    class G:
        SPLIT_RATIO = 0.8
        WINDOW_SIZE = 30
        BATCH_SIZE = 3
        NO_OF_FEATURES = 16

    # Create windowed datasets
    def windowed_dataset(series, date_column, window_size=G.WINDOW_SIZE):
        X = []

        for i in range(G.NO_OF_FEATURES):
            X.append([])
            for j in range(window_size, series.shape[0]):
                X[i].append(series[j - window_size:j, i])

        # Arrange the axes to be no_of_samples, window_size, no_of_features
        X = np.moveaxis(X, [0], [2])

        X = np.array(X)

        y = series[window_size:, -1]
        y = np.array(y)
        y = np.reshape(y, (len(y), 1))

        date_column = date_column[window_size:].values

        return X, y, date_column
    
    # Split data to train and test
    def train_test_split(X, y, split):

        split_time = int(split * len(X))
        X_train = X[:split_time]
        y_train = y[:split_time]
        X_test = X[split_time:]
        y_test = y[split_time:]

        return X_train, y_train, X_test, y_test
    
    # Download data from Yahoo Finance
    data = yf.download(tickers='^NDX', start='1985-10-01', end='2023-12-31')

    data['RSIF'] = ta.rsi(data.Close, length=14)
    data['RSIS'] = ta.rsi(data.Close, length=25)
    data['EMAF'] = ta.ema(data.Close, length=20)
    data['EMAM'] = ta.ema(data.Close, length=100)
    data['EMAS'] = ta.ema(data.Close, length=200)
    data['VWAP'] = ta.vwap(data.High, data.Low, data.Close, data.Volume, length=14)
    macd_data = ta.macd(data.Close)
    data['MACD_Line'] = macd_data['MACD_12_26_9']
    data['MACD_Signal_Line'] = macd_data['MACDs_12_26_9']
    data['MACD_Histogram'] = macd_data['MACDh_12_26_9']
    stoch_oscill_data = ta.stoch(data.High, data.Low, data.Close)
    data['STOCHk_14_3_3'] = stoch_oscill_data['STOCHk_14_3_3']
    data['STOCHd_14_3_3'] = stoch_oscill_data['STOCHd_14_3_3']

    data['NextClosing'] = data['Adj Close'].shift(-1)

    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    date_column = data['Date']
    data.drop(['Close', 'Date'], axis=1, inplace=True)
    dataset = data

    print(len(dataset.columns))

    dataset_array = dataset.values  
    X, y, windowed_date_column= windowed_dataset(dataset_array, date_column)

    print(X.shape)
    print(y.shape)

    # Rescale X's individual columns seperately 
    individual_scalers = []
    for feature_no in range(X.shape[2]):
        feature_column = X[:, :, feature_no].reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        X[:, :, feature_no] = scaler.fit_transform(feature_column).reshape(X[:, :, feature_no].shape)
        individual_scalers.append(scaler)

    # Rescale y
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = y_scaler.fit_transform(y)

    X_train, y_train, X_test, y_test = train_test_split(X, y_scaled, G.SPLIT_RATIO)
    date_test = windowed_date_column[int(G.SPLIT_RATIO * len(windowed_date_column)):]

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # Create model
    def create_uncompiled_model():

        model = tf.keras.models.Sequential([ 
            tf.keras.layers.LSTM(150, activation='linear', input_shape=(G.WINDOW_SIZE, G.NO_OF_FEATURES)),
            tf.keras.layers.Dense(50, activation='linear'),
            tf.keras.layers.Dense(1, activation='linear')
        ]) 

        return model

    def create_model():
        model = create_uncompiled_model()

        model.compile(loss=tf.keras.losses.Huber(),
                        optimizer=tf.keras.optimizers.SGD(learning_rate = 0.055, momentum = 0.9),
                        metrics=["mse"])  

        return model

    model = create_model()
    history = model.fit(X_train, y_train, epochs=50, batch_size=G.BATCH_SIZE)

    # Save model 
    model.save("stock_price_prediction_model_v1.keras")

    def compute_metrics(true_series, forecast):

        mse = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
        mae = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()

        return mse, mae

    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    y_true = y_scaler.inverse_transform(y_test)

    mse, mae = compute_metrics(y_true, y_pred)

    overall_avg_mse = np.mean(mse)
    overall_avg_mae = np.mean(mae)

    print("Overall Average MSE:", overall_avg_mse)
    print("Overall Average MAE:", overall_avg_mae)

    plt.figure(figsize=(10, 6))
    plt.plot(date_test, y_true, label='True Values', color='black', linestyle='-')
    plt.plot(date_test, y_pred, label='Predicted Values', color='green', linestyle='-')
    plt.xlabel('Date')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()




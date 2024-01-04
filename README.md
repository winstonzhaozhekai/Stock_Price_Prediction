# Stock Price Prediction with LSTM

## Overview
This project aims to predict stock prices using a Long Short-Term Memory (LSTM) neural network. The model is trained on historical stock data along with numerous indicators and evaluates its performance on a test set. The implementation is done in Python using various libraries, such as TensorFlow, Keras, NumPy, pandas, Matplotlib, and others.

## Dataset
Historical stock data are all obtained from Yahoo Finance

## Requirements
Make sure you have the following dependencies installed:

* Python
* TensorFlow
* NumPy
* pandas
* scikit-learn
* matplotlib
* yfinance
* pandas_ta

You can install the required packages using the following command:
```
pip install tensorflow numpy pandas scikit-learn matplotlib yfinance pandas_ta
```
## Project Structure
- **Imports**: Importing necessary libraries for data processing, machine learning, and visualisation.
- **Configuration**: Setting up configuration parameters, such as random seed and data class for global constants
- **Data Processing**: Downloading historical stock data from Yahoo Finance, adding technical indicators, and preparing the dataset for training and testing.
- **Data Splitting**: Splitting the dataset into training and testing sets.
- **Scaling**: Scaling data before training.
- **Model Creation**: Defining and training the LSTM neural network model for stock price prediction.
- **Model Evaluation**: Evaluating the model on the test set and computing metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- **Results Visualisation**: Plotting the true and predicted stock prices for visual inspection.

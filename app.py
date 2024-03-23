import numpy as np
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(
    page_title='Stock Trend Prediction',
    page_icon=':chart_with_upwards_trend:',
    layout='wide'
)

# Set up CSS styles
st.markdown(
    """
    <style>
    .title {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .subheader {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    .text-input {
        font-size: 18px;
        padding: 10px;
    }
    
    .chart {
        width: 100%;
        height: 400px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    .predicted-price {
        font-size: 24px;
        font-weight: bold;
        color: #3366ff;
    }
    
    .button-container {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set page title and button
st.markdown("<h1 class='title'>Stock Trend Prediction</h1>", unsafe_allow_html=True)

# Load the data
yfin.pdr_override()
start = '2010-01-01'
end = date.today().strftime('%Y-%m-%d')

# Load the tickers
def load_tickers(file_path):
    tickers = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            ticker, company = line.strip().split(',')
            tickers[ticker] = company
    return tickers

tickers = load_tickers('tickers.txt')

# Get user input
user_input = st.text_input('Enter Stock Ticker or Company Name', 'TSLA', key='stock-ticker')

# Filter tickers based on user input
filtered_tickers = [ticker for ticker in tickers if user_input.lower() in ticker.lower() or user_input.lower() in tickers[ticker].lower()]

# Display dropdown suggestions if user input is not empty
if user_input:
    st.markdown("<h4>Suggestions:</h4>", unsafe_allow_html=True)
    st.markdown("<ul>" + "".join([f"<li>{ticker}: {tickers[ticker]}</li>" for ticker in filtered_tickers]) + "</ul>", unsafe_allow_html=True)

# Fetch stock data
if user_input in tickers:
    ticker_symbol = user_input
else:
    ticker_symbol = filtered_tickers[0] if filtered_tickers else None

if ticker_symbol:
    df = pdr.get_data_yahoo(ticker_symbol, start, end)

    # Describe data
    st.subheader('Data from 2010 - Current')
    st.write(df.describe())

    # Visualize closing price vs time
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)

    # Moving averages
    st.subheader('Closing Price vs Time chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(df.Close, 'b')
    st.pyplot(fig)

    #Split data into training and testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    #Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    #Load the model
    model = load_model('keras_model.h5')

    #Prepare testing data
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
      x_test.append(input_data[i-100: i])
      y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    #Make predictions
    y_predicted = model.predict(x_test)

    #Scale back the predicted and test data
    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    #Plot predictions vs original
    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)







    #Create a new scaler object
    scaler_new = MinMaxScaler(feature_range=(0, 1))

    #Fit the scaler to the entire dataset
    data_scaled = scaler_new.fit_transform(df['Close'].values.reshape(-1, 1))

    #Preprocess data for prediction
    last_100_days = data_scaled[-100:]
    input_data = scaler_new.transform(last_100_days)

    x_input = np.array(input_data).reshape(1, 100, 1)

    #Make prediction
    predicted_price = model.predict(x_input)
    predicted_price = predicted_price[0][0] / scale_factor

    #Get today's date
    today = date.today()

    # Get today's price
    today_price = df['Close'].iloc[-1]

    # Get the whole number part of today's price
    today_whole_number = int(today_price)
    adjusted_price = predicted_price
    # Get the whole number part of the adjusted price
    adjusted_whole_number = int(adjusted_price)

    # Count the number of digits in the whole number part of today's price
    digits_today = len(str(today_whole_number))

    # Count the number of digits in the whole number part of the adjusted price
    digits_adjusted = len(str(adjusted_whole_number))

    # Calculate the adjustment factor based on the difference in digits
    adjustment_factor = 10 ** (digits_today - digits_adjusted)

    # Adjust the predicted price by multiplying it with the adjustment factor
    adjusted_price *= adjustment_factor

    # Check if the adjusted price whole number has the first digit in the range 1-9
    if adjusted_whole_number < 1 or adjusted_whole_number > 9:
        # Adjust the adjusted price whole number to the range 1-9
        adjusted_price /= 10

    # Adjust the number of digits in the adjusted price whole number to match today's price whole number
    while len(str(int(adjusted_price))) < digits_today:
        adjusted_price *= 10

    #Get tomorrow's date
    tomorrow = today + pd.DateOffset(days=1)

    #Display the predicted price for tomorrow
    st.subheader('Next Day Prediction')
    st.markdown(
        f"Date: {tomorrow}, Predicted Price: <span class='predicted-price'>{adjusted_price}</span>",
        unsafe_allow_html=True
    )
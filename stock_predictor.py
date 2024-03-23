import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yahoo_fin.stock_info as si
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Input
from datetime import datetime

def predict_next_n_days(ticker, n, model='LSTM', n_steps=30, epochs=20, batch_size=32):

  # Download data and convert index to datetime
  ticker_data = si.get_data(ticker, start_date="2014-01-01")
  ticker_data.index = pd.to_datetime(ticker_data.index)

  # Preprocess data
  data = ticker_data['close'].values.reshape(-1, 1)
  scaler = MinMaxScaler(feature_range=(0, 1))  # Adjusted scaling range to avoid overflow
  data_scaled = scaler.fit_transform(data)

  # Prepare data for LSTM
  x, y = [], []
    
  for i in range(len(data_scaled) - n_steps):
    x.append(data_scaled[i:i + n_steps, 0])
    y.append(data_scaled[i + n_steps, 0])
      
  x, y = np.array(x), np.array(y)
  x = x.reshape((x.shape[0], x.shape[1], 1))

  # Build the model (with Input layer)
  if model == 'LSTM':
        model_rnn = Sequential()
        model_rnn.add(LSTM(units=50, activation='relu', input_shape=(n_steps, 1)))
        model_rnn.add(Dense(units=1))
      
  elif model == 'GRU':
        model_rnn = Sequential()
        model_rnn.add(GRU(units=50, activation='relu', input_shape=(n_steps, 1)))
        model_rnn.add(Dense(units=1))
        
  else:
    raise ValueError("Invalid model type. Choose 'LSTM' or 'GRU'.")

  model_rnn.compile(optimizer='adam', loss='mean_squared_error')

  # Train the model
  model_rnn.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)

  # Make predictions for the next n days
  input_sequence = data_scaled[-n_steps:].reshape((1, n_steps, 1))
  predictions = []

  for _ in range(n):
    pred = model_rnn.predict(input_sequence, verbose=0)
    predictions.append(pred[0, 0])
    input_sequence = np.append(input_sequence[:, 1:, :], pred.reshape((1, 1, 1)), axis=1)

  predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))



  # Create DataFrame for predicted prices
  predicted_prices_df = pd.DataFrame(predicted_prices, index=pd.date_range(start=datetime.today().date() + pd.offsets.BDay(), periods=n, freq='B'))
  predicted_prices_df.columns = ['Predicted']

  return predicted_prices_df, ticker_data, n



# def plot_predicted_prices(predicted_prices_df, ticker_data, ticker, filename="predicted_prices.png", days=n):


#   years_ago = datetime.today().date() - pd.DateOffset(years=1)
#   ticker_data_recent = ticker_data[ticker_data.index >= years_ago]  

#   plt.figure(figsize=(10, 6))

#   # Plot actual prices (last 2 years)
#   plt.plot(ticker_data_recent.index, ticker_data_recent['close'], label='Actual Prices', color='blue', linewidth=1)

#   # Plot predicted prices
#   plt.plot(predicted_prices_df.index, predicted_prices_df['Predicted'], label='Predicted Prices', color='orange', linewidth=2)

#   # Styling for a more visually appealing plot
#   plt.xlabel('Date', fontsize=12)
#   plt.ylabel('Price', fontsize=12)
#   plt.title(f'Predicted vs. Actual Prices for {ticker} for {days} days', fontsize=14)
#   plt.legend(fontsize=12)
#   plt.grid(True)
#   plt.tight_layout()

#   # Save the plot (replace "predicted_prices.png" with your desired filename)
#   plt.savefig(filename)

#   # Close the plot figure (optional)
#   plt.close()


def plot_predicted_prices(predicted_prices_df, ticker_data, ticker, days=30, filename=None):
    years_ago = datetime.today().date() - pd.DateOffset(years=1)
    ticker_data_recent = ticker_data[ticker_data.index >= years_ago]  

    plt.figure(figsize=(10, 6))

    # Plot actual prices (last 2 years)
    plt.plot(ticker_data_recent.index, ticker_data_recent['close'], label='Actual Prices', color='blue', linewidth=1)

    # Plot predicted prices
    plt.plot(predicted_prices_df.index, predicted_prices_df['Predicted'], label='Predicted Prices', color='orange', linewidth=2)

    # Styling for a more visually appealing plot
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.title(f'Predicted vs. Actual Prices for {ticker} for {days} days', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename)
        plt.close()
        return filename
    else:
        # Convert plot to base64 and return
        import io
        import base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{plot_data}"
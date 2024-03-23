# Stock Prediction Web App

This is a Flask web application for predicting and visualizing stock prices using machine learning models. Users can select a stock ticker symbol, specify the number of days for prediction, and choose between different machine learning models (LSTM or GRU). The predicted prices are displayed on the web page along with the actual prices for comparison.

# Directory Structure

The project directory is organized as follows:

```plaintext
stock_prediction_app/
├── app.py                   # Flask application file
├── stock_predictor.py       # Module for predicting and plotting stock prices
├── stock_prediction.ipynb   # Jupyter Notebook for experimenting and testing code
├── static/                  # Directory for static files
│   └── images/              # Directory to store generated plot images
├── templates/               # Directory for HTML templates
│   └── index.html           # HTML template for the main page
└── requirements.txt         # File containing all Python dependencies

```

# Getting Started

To run the application, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the project directory.
```plaintext
cd stock_prediction_app
```
3. Install the required Python packages using pip.
```plaintext
pip install -r requirements.txt
```
4. Run the Flask application.
```plaintext
python app.py
```
5. Open a web browser and go to http://127.0.0.1:5000 to access the application.

# Usage

- Select a stock ticker symbol from the dropdown menu.
- Specify the number of days for prediction.
- Choose a machine learning model (LSTM or GRU).
- Click on the "Predict" button.
- The predicted stock prices will be displayed on the web page along with the actual prices.


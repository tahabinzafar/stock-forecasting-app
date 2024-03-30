from flask import Flask, render_template, request
from stock_predictor import predict_next_n_days, plot_predicted_prices
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        n_days = int(request.form['n_days'])
        model = request.form['model']
        
        # Predict the next n days of closing prices
        predicted_prices_df, ticker_data, days = predict_next_n_days(ticker, n_days, model=model)

        # Plot the predicted prices and get the filename
        plot_file = plot_predicted_prices(predicted_prices_df, ticker_data, ticker, days=days, filename='static/images/predicted_prices.png')

        return render_template('index.html', plot_image=plot_file)
    
    return render_template('index.html', plot_image=None)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)

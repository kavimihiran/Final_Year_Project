import yfinance as yf

# Fetch historical BTC data from Yahoo Finance
btc = yf.download('BTC-USD', start='2014-01-26', end='2024-01-31')

# Select columns for 'Open', 'High', 'Low', 'Close'
btc_prices = btc[['Open', 'High', 'Low', 'Close']]

# Reset the index and create a 'Date' column
btc_prices.reset_index(inplace=True)

# Save the dataset to a CSV file
btc_prices.to_csv('dataset/btc_2014_to_2024.csv', index=False)


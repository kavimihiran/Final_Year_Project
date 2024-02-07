import pandas as pd

# Read BTC.csv containing Bitcoin prices
btc_data = pd.read_csv('dataset/btc_2014_to_2024.csv')

# Read the sentiment data with sentiment scores
sentiment_data = pd.read_csv('combined_tweets_with_sentiment.csv')

sentiment_data['Date'] = pd.to_datetime(sentiment_data['created_at'], format="%Y-%m-%dT%H:%M:%S.%fZ", errors='coerce')
sentiment_data['Date'] = sentiment_data['Date'].fillna(pd.to_datetime(sentiment_data['created_at'], errors='coerce'))
sentiment_data['Date'] = sentiment_data['Date'].dt.date

# Convert 'Date' column in btc_data to date format
btc_data['Date'] = pd.to_datetime(btc_data['Date']).dt.date

# Merge BTC data with sentiment data based on the 'Date' column
merged_data = pd.merge(btc_data, sentiment_data[['Sentiment_Score', 'Date']], on='Date', how='left')

# Save the merged data (BTC.csv with added sentiment scores) to a new file
merged_data.to_csv('datasets_with_sentiment_Score/bitcoin_sentiments_usable_filled.csv', index=False)





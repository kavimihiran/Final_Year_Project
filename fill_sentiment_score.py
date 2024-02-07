import pandas as pd

# Read the CSV file
fill_sentiment = pd.read_csv('XRP_with_sentiment.csv')

# Sort the data by date
fill_sentiment = fill_sentiment.sort_values('Date')

# Fill the first missing value with the first valid value in the column
fill_sentiment['Sentiment_Score'] = fill_sentiment['Sentiment_Score'].fillna(method='ffill')

# Fill the missing values in 'sentiment_score' column with the average of the past five values
fill_sentiment['Sentiment_Score'] = fill_sentiment['Sentiment_Score'].fillna(fill_sentiment['Sentiment_Score'].rolling(10, min_periods=1).mean())

# Fill any remaining missing values with 0.0
fill_sentiment['Sentiment_Score'] = fill_sentiment['Sentiment_Score'].fillna(0.0)

# Save the filled data to a new CSV file
fill_sentiment.to_csv('datasets_with_sentiment_Score/Ripple_sentiments_usable_filled.csv', index=False)
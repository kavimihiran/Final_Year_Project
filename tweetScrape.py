import snscrape.modules.twitter as sntwitter
import pandas as pd
import time

# Define the search query
query = "Bitcoin OR #Bitcoin OR BTC OR #BTC since:2014-01-01 until:2024-01-01"

tweets = []

# Function to write tweets to CSV file
def write_tweets_to_csv(tweets_list):
    df = pd.DataFrame(tweets_list, columns=['Date', 'User', 'Tweet'])
    df.to_csv('final_dataset/tweets_snscrape_12to18.csv', mode='a', index=False, header=False)

def search_tweets():
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        print(i)
        try:
            tweets.append([tweet.date, tweet.user.username, tweet.rawContent])
            write_tweets_to_csv([[tweet.date, tweet.user.username, tweet.rawContent]])
        except Exception as e:
            print(f"Error scraping tweet {i}: {e}")

# Call the function to start tweet retrieval
search_tweets()
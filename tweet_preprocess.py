import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


sentiment_data = pd.read_csv('final_dataset/tweets_snscrape_12to18.csv')

# Function for tweet preprocessing
def preprocess_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+|@\S+|[^\w\s]", "", tweet)
    tokens = word_tokenize(tweet)
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Preprocess tweets from sentiment_data and perform sentiment analysis
sentiment_data['Processed_Tweet'] = sentiment_data['Tweet'].apply(preprocess_tweet)

# sentiment_data = remove_duplicate_rows(sentiment_data)

sentiment_data.to_csv('tweetsData/Final_BTC_tweets_Preprocessed.csv', index=False)


import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiment_data_preprocessed = pd.read_csv('tweetsData/Final_XRP_tweets_Preprocessed.csv')
#sentiment analysis using vader
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(tweet):
    # Use the VADER sentiment analyzer to get the polarity scores
    scores = analyzer.polarity_scores(tweet)
    # Return the compound score, which ranges from -1 (very negative) to +1 (very positive)
    return scores['compound']

# sentiment_data['Sentiment_Score'] = sentiment_data['Processed_Tweet'].apply(analyze_sentiment)
sentiment_data_preprocessed['Processed_Tweet'] = sentiment_data_preprocessed['Processed_Tweet'].fillna('')
sentiment_data_preprocessed['Sentiment_Score'] = sentiment_data_preprocessed['Processed_Tweet'].apply(get_sentiment_score)

sentiment_data_preprocessed.to_csv('combined_tweets_with_sentiment.csv',index=False)


# Importing necessary libraries
import nltk # Natural Language Toolkit for python
from nltk.sentiment import SentimentIntensityAnalyzer # VADER sentiment analysis tool (Valence Aware Dictionary and sEntiment Reasoner)

# Download the vader_lexicon package
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    # Initialize the VADER sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()

    # Compute and print the sentiment scores
    sentiment = sia.polarity_scores(text)
    print(sentiment)

# Test the function with a sample text
analyze_sentiment("NLTK is a great library for Natural Language Processing!")
analyze_sentiment("I love this product! It's amazing.")
analyze_sentiment("This is the worst experience I've ever had.")
analyze_sentiment("It's okay, nothing special")
analyze_sentiment("Meh, could be better.")
analyze_sentiment("Oh, great. Another meeting. That's exactly what I needed.")
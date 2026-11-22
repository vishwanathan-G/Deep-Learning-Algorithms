# classifier.py - Sentiment, Topic, Spam detection
# Saved from cell 8 of nlp.ipynb, as planned in the notebook's project layout,
# so that main.py can import it. The code below is the notebook cell's code.

#!pip install TextBlob
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def sentiment_analysis(text):
    blob = TextBlob(text)
    print("Sentiment Polarity:", blob.sentiment.polarity)

def spam_detection():
    # Sample
    texts = ['win money now', 'hello friend', 'buy now', 'how are you?']
    labels = [1, 0, 1, 0]  # 1 = spam, 0 = not spam

    vect = CountVectorizer()
    X = vect.fit_transform(texts)
    model = MultinomialNB()
    model.fit(X, labels)

    test = vect.transform(['win free prize'])
    print("Spam Prediction:", model.predict(test)[0])

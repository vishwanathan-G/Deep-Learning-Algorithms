# preprocess.py - Text Preprocessing
# Saved from cell 4 of nlp.ipynb, as planned in the notebook's project layout,
# so that main.py can import it. The code below is the notebook cell's code.

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')  # tokenizer tables needed by NLTK 3.9 and later
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) #Removes all symbols except words (\w) and whitespace (\s).

    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english')) # Removes common but useless words like:
                                                 #"is", "a", "the", "on", etc.
    filtered = [w for w in tokens if w not in stop_words]

    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in filtered]

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]

    return lemmatized

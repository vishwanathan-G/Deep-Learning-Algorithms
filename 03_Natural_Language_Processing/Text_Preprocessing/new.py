# ---------------------------
# NLP + TTS Processing Script
# ---------------------------

# Import necessary libraries
import nltk  # Natural Language Toolkit for text processing
import string  # For punctuation removal
import pyttsx3  # For offline text-to-speech conversion

# Import specific modules from NLTK
from nltk.corpus import stopwords  # List of common stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer  # Tools for word reduction
from nltk import pos_tag, word_tokenize, ne_chunk  # POS tagging and NER
from nltk.tree import Tree  # Tree data structure used in NER

# -----------------------------
# Download required NLTK data
# -----------------------------
nltk.download('punkt')  # Tokenizer
nltk.download('stopwords')  # List of stopwords
nltk.download('averaged_perceptron_tagger_eng')  # POS tagger model
nltk.download('maxent_ne_chunker')  # Named entity chunker
nltk.download('maxent_ne_chunker_tab')  # Chunker tab support
nltk.download('words')  # English word list for NER
nltk.download('wordnet')  # WordNet dictionary for lemmatization

# -----------------------------
# Initialize tools
# -----------------------------
stop_words = set(stopwords.words('english'))  # Set of common stopwords to ignore
stemmer = PorterStemmer()  # Create stemming object
lemmatizer = WordNetLemmatizer()  # Create lemmatizer object
engine = pyttsx3.init()  # Initialize text-to-speech engine

# -----------------------------
# Take input from user
# -----------------------------
text = input("Enter your text: ")  # Get user input for processing

# -----------------------------
# Basic Text Statistics
# -----------------------------
print("\n---- Text Statistics ----")
print("Word Count:", len(word_tokenize(text)))  # Count total words
print("Sentence Count:", len(nltk.sent_tokenize(text)))  # Count total sentences

# -----------------------------
# Tokenization: Split text into words
# -----------------------------
tokens = word_tokenize(text)
print("\n---- Tokens ----")
print(tokens)

# -----------------------------
# Stopword Removal: Filter out common and punctuation words
# -----------------------------
filtered_tokens = [
    word for word in tokens
    if word.lower() not in stop_words and word not in string.punctuation
]
print("\n---- After Stopword Removal ----")
print(filtered_tokens)

# -----------------------------
# Stemming: Reduce words to root forms (may not be real words)
# -----------------------------
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
print("\n---- Stemming ----")
print(stemmed_words)

# -----------------------------
# Lemmatization: Reduce words to base dictionary form
# -----------------------------
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\n---- Lemmatization ----")
print(lemmatized_words)

# -----------------------------
# POS Tagging: Identify grammatical roles (noun, verb, etc.)
# -----------------------------
pos_tags = pos_tag(filtered_tokens, lang='eng')
print("\n---- Part of Speech (POS) Tagging ----")
print(pos_tags)

# -----------------------------
# Named Entity Recognition: Find names of people, places, orgs
# -----------------------------
print("\n---- Named Entities ----")
chunks = ne_chunk(pos_tags)  # Break into named entity trees
named_entities = []

# Traverse each chunk to extract named entities
for chunk in chunks:
    if isinstance(chunk, Tree):  # If it’s a named entity
        entity = " ".join(c[0] for c in chunk)  # Combine words into a phrase
        named_entities.append(entity)

print(named_entities)

# -----------------------------
# Text-to-Speech: Read out processed text
# -----------------------------
print("\nReading Processed Text...")
final_text = ' '.join(lemmatized_words)  # Join processed words into a sentence
engine.say(final_text)  # Speak the text
engine.runAndWait()  # Wait until speaking is finished

# -----------------------------
# End of Program
# -----------------------------
print("\nProcess Complete.")

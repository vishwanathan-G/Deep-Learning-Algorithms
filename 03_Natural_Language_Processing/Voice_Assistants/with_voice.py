import nltk
import string
import pyttsx3
import speech_recognition as sr
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, word_tokenize, ne_chunk
from nltk.tree import Tree

# Download necessary resources
nltk.download('punkt')
nltk.download('punkt_tab')  # Tokenizer tables needed by NLTK 3.9 and later
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('wordnet')

# Initialize modules
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
engine = pyttsx3.init()

# --- Voice Input Section ---
print("🎙️ Speak now...")

recognizer = sr.Recognizer()
with sr.Microphone() as source:
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)

try:
    text = recognizer.recognize_google(audio)
    print("\n✅ You said:", text)
except Exception as e:
    print("❌ Could not recognize your voice:", e)
    exit()

# --- NLP Processing Section ---

# Text Statistics
print("\n---- Text Statistics ----")
print("Word Count:", len(word_tokenize(text)))
print("Sentence Count:", len(nltk.sent_tokenize(text)))

# Tokenization
tokens = word_tokenize(text)
print("\n---- Tokens ----")
print(tokens)

# Stopword Removal
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
print("\n---- After Stopword Removal ----")
print(filtered_tokens)

# Stemming
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
print("\n---- Stemming ----")
print(stemmed_words)

# Lemmatization
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\n---- Lemmatization ----")
print(lemmatized_words)

# POS Tagging
pos_tags = pos_tag(filtered_tokens, lang='eng')
print("\n---- Part of Speech (POS) Tagging ----")
print(pos_tags)

# Named Entity Recognition (NER)
print("\n---- Named Entities ----")
chunks = ne_chunk(pos_tags)
named_entities = []
for chunk in chunks:
    if isinstance(chunk, Tree):
        entity = " ".join(c[0] for c in chunk)
        named_entities.append(entity)
print(named_entities)

# Text to Speech Output
print("\nReading Processed Text...")
final_text = ' '.join(lemmatized_words)
engine.say(final_text)
engine.runAndWait()

print("\nProcess Complete.")

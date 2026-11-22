# Text Preprocessing - spaCy and NLTK Pipelines

Two command-line scripts that take English text typed by the user and show the standard preprocessing and annotation steps of an NLP pipeline, one with spaCy and one with NLTK.

## Files in this folder

| File | Description |
|---|---|
| [`code1.py`](code1.py) | Interactive spaCy analyser. Reads sentences in a loop until `quit` or `exit` and prints a token table (token, Porter stem, lemma, POS, fine-grained tag, stop-word flag, dependency label), the words left after stop-word and punctuation removal, lemmatized and stemmed text, named entities and sentence boundaries. |
| [`new.py`](new.py) | NLTK pipeline for one line of input: word and sentence counts, tokens, stop-word removal, Porter stemming, WordNet lemmatization, POS tagging and named-entity chunking. It then reads the lemmatized words aloud with `pyttsx3`. Downloads the NLTK data it needs at start-up. |

## 1. Algorithm Name

Text preprocessing and linguistic annotation:

- Tokenization and sentence segmentation
- Stop-word (and punctuation) removal
- Stemming (Porter stemmer)
- Lemmatization (spaCy lemmatizer in `code1.py`, WordNet lemmatizer in `new.py`)
- Part-of-speech (POS) tagging
- Named-entity recognition (NER)
- Dependency parsing (`code1.py`)
- Text-to-speech output of the processed text (`new.py`)

## 2. Definition

Text preprocessing turns raw text into clean, consistent units that later NLP steps can use. **Tokenization** splits text into words and punctuation; **sentence segmentation** splits it into sentences. **Stop-word removal** drops very common function words. **Stemming** cuts words down to a crude root with rules, while **lemmatization** maps each word to its dictionary form. **POS tagging** labels each token with its grammatical category, **NER** finds names of people, places, organisations, dates and similar spans, and **dependency parsing** links each word to the word it grammatically depends on.

## 3. How It Works

**`code1.py` (spaCy)**

1. Loads the `en_core_web_sm` pipeline. If the model is missing, the script exits with the command to install it.
2. Asks for a sentence in a loop. `quit` or `exit` stops the program; empty input asks again.
3. `nlp(text)` runs spaCy's tokenizer, tagger, dependency parser, lemmatizer and entity recognizer in one pass.
4. Prints one row per token (whitespace tokens are skipped): the token, its Porter stem (from NLTK, on the lower-cased token), lemma, coarse POS, fine-grained tag, whether it is a stop word, and its dependency label.
5. Keeps tokens that are not stop words, punctuation or whitespace, and prints them, their lower-cased lemmas and their stems.
6. Prints each named entity with its label, or `No named entities found.`
7. Prints each sentence found by the parser.

**`new.py` (NLTK)**

1. Calls `nltk.download` for `punkt`, `punkt_tab`, `stopwords`, `averaged_perceptron_tagger_eng`, `maxent_ne_chunker`, `maxent_ne_chunker_tab`, `words` and `wordnet`.
2. Creates the English stop-word set, a `PorterStemmer`, a `WordNetLemmatizer` and a `pyttsx3` speech engine.
3. Reads one line with `input("Enter your text: ")`.
4. Prints the word count (`len(word_tokenize(text))`, which also counts punctuation tokens) and the sentence count (`len(nltk.sent_tokenize(text))`).
5. Tokenizes with `word_tokenize` and prints the tokens.
6. Removes tokens whose lower-case form is an English stop word or that are a punctuation character from `string.punctuation`.
7. Stems and lemmatizes the remaining tokens.
8. POS-tags the remaining tokens with `pos_tag(filtered_tokens, lang='eng')`.
9. Runs `ne_chunk` on those tags and collects the words of every entity subtree as a phrase (only the entity text is printed, not its label).
10. Joins the lemmatized words and speaks them with `engine.say` and `engine.runAndWait`.

## 4. Why It Is Used

- Machine-learning models need text as discrete, consistent units; tokenization provides them.
- Stop-word removal, stemming and lemmatization shrink the vocabulary, so related word forms ("connects", "connected", "connecting") count as one feature.
- POS tags, entities and dependency relations add structure that downstream tasks such as information extraction, question answering and chatbots rely on.

## 5. Mathematical Intuition

**Why stop words are removed.** Word frequencies in natural language follow a power law: a few words (mostly function words such as "the", "of", "is") account for a large share of all tokens while carrying little topical information. Removing them shrinks the data with little loss for tasks such as search and topic classification.

**Porter stemming** is rule-based, not statistical. A word is viewed as $[C](VC)^m[V]$, where $C$ and $V$ are runs of consonants and vowels and $m$ is the word's "measure". Suffix rules fire only when a condition on $m$ holds, for example `(m > 0) EED -> EE` turns "agreed" into "agree" but leaves "feed" unchanged.

**Lemmatization** is a lookup: $(\text{word}, \text{POS}) \rightarrow \text{lemma}$ using a dictionary (WordNet in `new.py`) plus inflection rules. Without a POS the WordNet lemmatizer assumes a noun.

**POS tagging** is sequence labelling: choose the tag sequence that scores best for the words,

```math
\hat{t}_{1..n} = \arg\max_{t_{1..n}} \; \text{score}(t_{1..n} \mid w_{1..n})
```

NLTK's averaged perceptron scores each tag with a weight vector and context features, $\text{score} = \mathbf{w} \cdot \phi(\text{context}, t)$; spaCy uses a neural network that predicts a tag for each token from its context.

**NER with NLTK's chunker** uses a maximum-entropy classifier over features $f_i$ of a word and its neighbours:

```math
P(\text{label} \mid \text{features}) = \frac{\exp\left(\sum_i \lambda_i f_i\right)}{Z}
```

**Dependency parsing** produces a tree in which every word has exactly one head and a relation label; spaCy builds it with a transition-based neural parser.

## 6. Important Parameters

**General parameters**

| Parameter | Meaning |
|---|---|
| Language model / resources | Which trained pipeline or data packages are used (for spaCy: `sm`, `md`, `lg`, `trf` sizes) |
| Stop-word list | Which words are treated as stop words |
| Stemmer | Porter, Snowball or Lancaster (more or less aggressive) |
| Lemmatizer POS | Part of speech passed to the lemmatizer |
| Case handling | Lower-casing before comparison or stemming |
| Punctuation handling | Whether punctuation tokens are kept |
| Text-to-speech settings | Voice, speaking rate and volume |

**Used in this code**

| Setting | `code1.py` | `new.py` |
|---|---|---|
| Library | spaCy (+ NLTK `PorterStemmer`) | NLTK |
| Model / data | `en_core_web_sm` | `punkt`, `punkt_tab`, `stopwords`, `averaged_perceptron_tagger_eng`, `maxent_ne_chunker`, `maxent_ne_chunker_tab`, `words`, `wordnet` |
| Stop words | spaCy `token.is_stop` | `stopwords.words('english')` |
| Punctuation filter | `token.is_punct` (whitespace via `token.is_space`) | token in `string.punctuation` |
| Stemmer | `PorterStemmer` on the lower-cased token | `PorterStemmer` on the filtered token |
| Lemmatizer | spaCy `token.lemma_`, lower-cased in the filtered output | `WordNetLemmatizer().lemmatize(word)` with its default noun POS |
| POS tagging | `token.pos_` and `token.tag_` on the full sentence | `pos_tag(filtered_tokens, lang='eng')` on the stop-word-filtered tokens |
| NER | `doc.ents` with labels | `ne_chunk(pos_tags)`, entity text only |
| Input | Loop; `quit` or `exit` ends | One line |
| Speech output | None | `pyttsx3` default voice and rate |

## 7. Advantages

- Smaller vocabulary and less noise, which helps classical models such as Naive Bayes or TF-IDF classifiers.
- Stemming is fast and needs no dictionary.
- Lemmatization returns real words, which are easier to read and interpret.
- POS tags, entities and dependencies give structure for rule-based and statistical extraction.
- spaCy runs every step in one call; NLTK exposes each step separately, which is useful for learning.

## 8. Disadvantages

- Stemming can over-cut ("university" and "universe" share a stem) or produce non-words.
- Stop-word lists include negations such as "not" and "no"; removing them can reverse the meaning for sentiment tasks.
- Tagging or chunking text after stop words have been removed (as `new.py` does) gives the tagger less context than the original sentence.
- The WordNet lemmatizer leaves verbs unchanged unless it is told the POS.
- Pretrained taggers and entity recognizers make mistakes on informal text, other domains and other languages.
- Modern transformer models use their own subword tokenizers, so heavy preprocessing is often unnecessary for them.

## 9. Real-World Applications

- Indexing documents for search engines
- Feature preparation for spam, sentiment and topic classifiers
- Information extraction (people, organisations, dates, amounts) from documents
- Chatbots and voice assistants that need to understand user requests
- Accessibility tools that read processed text aloud

## 10. Python Libraries Used

| Library | Imports | Used for |
|---|---|---|
| spaCy | `import spacy` (`code1.py`) | Tokenization, POS, lemmas, dependencies, entities, sentences |
| NLTK | `from nltk.stem import PorterStemmer` (`code1.py`); `import nltk`, `from nltk.corpus import stopwords`, `from nltk.stem import PorterStemmer, WordNetLemmatizer`, `from nltk import pos_tag, word_tokenize, ne_chunk`, `from nltk.tree import Tree` (`new.py`) | Stemming, tokenization, stop words, lemmatization, POS tagging, NE chunking |
| pyttsx3 | `import pyttsx3` (`new.py`) | Offline text-to-speech |
| string (standard library) | `import string` (`new.py`) | Punctuation list |

## 11. Dataset Information

There is no dataset file. Both scripts process text typed by the user at run time. They rely on pretrained resources:

- **spaCy `en_core_web_sm`**: a small English pipeline (tagger, parser, lemmatizer, entity recognizer) installed with `python -m spacy download en_core_web_sm`.
- **NLTK data packages** downloaded by `new.py` (stop-word lists, Punkt sentence tokenizer with its `punkt_tab` tables, perceptron POS tagger, maximum-entropy NE chunker, English word list and WordNet). By default they are stored in `%APPDATA%\nltk_data`.

## 12. How to Execute

From the repository root, after following "Getting started" in the [root README](../../README.md):

```powershell
cd 03_Natural_Language_Processing\Text_Preprocessing

# once, for code1.py
python -m spacy download en_core_web_sm

python code1.py
python new.py
```

Notes:

- `new.py` calls `nltk.download` every time it starts. The first run needs internet access; later runs report the packages as up to date.
- `new.py` speaks through the operating system's speech engine (SAPI5 on Windows), so turn on speakers or headphones. No internet is needed for speech.
- `new.py` downloads `punkt_tab` as well as `punkt`. NLTK 3.9 and later load `punkt_tab` for `word_tokenize` and `sent_tokenize`; with NLTK 3.10.3, `punkt` alone is not enough and tokenization raises a `LookupError` for `punkt_tab`. The `punkt_tab` download was added to the script on 22 November 2026.

## 13. Expected Output

There are no saved outputs for these scripts; this section describes what they print.

**`code1.py`**

1. `English NLP Analyzer` and short instructions, then the prompt `Enter text:`.
2. For each sentence entered, these blocks separated by lines of `=`:
   - `ORIGINAL TEXT` - the input.
   - `TOKEN-LEVEL ANALYSIS` - a table with the columns `TOKEN`, `STEM`, `LEMMA`, `POS`, `TAG`, `STOPWORD`, `DEPENDENCY`.
   - `STOPWORD REMOVAL` - `Words kept: [...]`.
   - `LEMMATIZED TEXT (stopwords removed)` and `STEMMED TEXT (stopwords removed)` - one line each.
   - `NAMED ENTITIES` - each entity text with its spaCy label (for example `PERSON`, `GPE`, `ORG`, `DATE`), or `No named entities found.`
   - `SENTENCES` - one line per sentence.
3. `Goodbye!` after `quit` or `exit`.

**`new.py`**

1. `[nltk_data]` download or "already up-to-date" messages.
2. The prompt `Enter your text:`.
3. Sections headed `---- Text Statistics ----` (word and sentence counts), `---- Tokens ----`, `---- After Stopword Removal ----`, `---- Stemming ----`, `---- Lemmatization ----` (each a Python list), `---- Part of Speech (POS) Tagging ----` (a list of `(word, tag)` tuples with tags such as `NN` for a noun, `VBD` for a past-tense verb and `JJ` for an adjective) and `---- Named Entities ----` (a list of entity phrases).
4. `Reading Processed Text...` while the lemmatized words are spoken, then `Process Complete.`

**Check on 22 November 2026** (Windows 11, NLTK 3.10.3):

- `new.py` ran end to end with the real NLTK on a clean NLTK data folder, with the speaker (`pyttsx3`) replaced by a test stand-in. Tokenization, stop-word removal, stemming, lemmatization, POS tags and named entities were produced, and the final text reached the speaker stand-in. `Voice_Assistants/with_voice.py`, which runs the same NLTK steps, was checked in the same way; for the sample sentence "Barack Obama was born in Hawaii. He was the president of the United States." the named entities were `['Barack', 'Obama', 'Hawaii', 'United States']`. The real speaker was not tested.
- `code1.py` was not run: spaCy could not load on the test machine because Windows Smart App Control blocked one of its compiled dependencies.

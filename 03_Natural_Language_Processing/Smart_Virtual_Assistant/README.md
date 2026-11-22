# Smart Virtual Assistant using NLP (Work in Progress)

A modular NLP assistant that is meant to take text or voice input, process it, extract information, understand it and respond by summarizing, translating, answering questions and speaking. The component functions are written in the notebooks; packaging them into the modules that `main.py` imports is still to be done.

## Files in this folder

| File | Description |
|---|---|
| [`nlp.ipynb`](nlp.ipynb) | "Smart Virtual Assistant using NLP": a short description, the planned module layout, one code cell per component (preprocessing, syntax, semantics, information extraction, classification, generation, speech, question answering), an integration cell and a `!pip install preprocess` cell. Saved outputs include errors (see "Current status"). |
| [`nlp1.ipynb`](nlp1.ipynb) | Earlier variant of the same component cells without the markdown. `parse_syntax` is defined twice, the integration cell imports `text_preprocessing` instead of `preprocess`, and a later cell runs `!pip install syntax_parse`. |
| [`main.py`](main.py) | Integration script. Imports the eight planned local modules and calls each component on one line of user input. |

## Current status

This project is a work in progress.

- `main.py` and the integration cell of `nlp.ipynb` import eight local modules - `preprocess`, `syntax_parse`, `semantic_analysis`, `info_extract`, `classifier`, `generation`, `speech_module`, `qa_module` - that are **not in this repository yet** (the integration cell of `nlp1.ipynb` imports `text_preprocessing` in place of `preprocess`). Their functions exist only as notebook cells, so the integration step does not run yet.
- The notebooks contain `!pip install preprocess` (`nlp.ipynb`) and `!pip install syntax_parse` (`nlp1.ipynb`). These commands look for packages on PyPI, not for the project's local modules. Saved output: `!pip install preprocess` installed an unrelated PyPI package named `preprocess` (version 2.0.0, with its dependency `future`), and the next run of the integration cell failed with `ImportError: cannot import name 'preprocess_text' from 'preprocess'`; `!pip install syntax_parse` ended with "No matching distribution found".
- Other saved errors are listed in section 13.

## 1. Algorithm Name

Modular NLP pipeline for a virtual assistant:

- Text preprocessing (lower-casing, punctuation removal, tokenization, stop-word removal, stemming, lemmatization)
- Syntactic analysis (POS tagging, dependency parsing)
- Semantic analysis (named-entity recognition; coreference is only simulated)
- Information extraction (entities and the sentence's root word)
- Text classification (lexicon-based sentiment polarity; Multinomial Naive Bayes spam detection)
- Text generation tasks (machine translation, LSA extractive summarization, transformer text generation)
- Extractive question answering (transformer pipeline)
- Speech recognition and text-to-speech

## 2. Definition

A virtual assistant is a system that accepts natural-language input (typed or spoken), works out what it says and what is being asked, and produces a useful response such as an answer, a summary, a translation or spoken output. Here the assistant is designed as a set of independent NLP components, each handling one stage, with an integration script that runs them in sequence.

## 3. How It Works

**Components defined in the notebooks** (the module column is the planned file name from the notebook's own layout):

| Planned module | Function(s) in the notebook | What the code does |
|---|---|---|
| `preprocess.py` | `preprocess_text(text)` | Lower-cases, removes every character that is not a word character or whitespace (`re.sub(r'[^\w\s]', '', text)`), tokenizes with `nltk.word_tokenize`, drops English stop words, applies `PorterStemmer`, then applies `WordNetLemmatizer` to the stems, and returns the list |
| `syntax_parse.py` | `parse_syntax(text)` | spaCy `en_core_web_sm`: prints each token with its POS tag, then `token --> dependency --> head` for each token |
| `semantic_analysis.py` | `analyze_semantics(text)` | Prints spaCy named entities with labels, then a fixed example line under "Coreference Resolution (Simulated)" |
| `info_extract.py` | `extract_entities_relations(text)` | Prints entities, then a "Relation" line for each token whose dependency label is `ROOT` |
| `classifier.py` | `sentiment_analysis(text)`, `spam_detection()` | TextBlob sentiment polarity; a `CountVectorizer` + `MultinomialNB` model trained on four sample sentences (`'win money now'`, `'hello friend'`, `'buy now'`, `'how are you?'` with labels 1, 0, 1, 0) that predicts the label of `'win free prize'` |
| `generation.py` | `translate_text(text, dest='fr')`, `summarize_text(text)`, `generate_text(prompt)` | `googletrans.Translator().translate`; sumy `LsaSummarizer` returning 2 sentences; `transformers.pipeline("text-generation")` with `max_length=30`. The cell also creates `pipeline("summarization")` at load time, which `summarize_text` does not use |
| `speech_module.py` | `recognize_speech()`, `speak_text(text)` | SpeechRecognition microphone capture with `recognize_google`; `pyttsx3` speech output |
| `qa_module.py` | `answer_question(question, context)` | `transformers.pipeline("question-answering")`, returns the answer span |

**Integration (`main.py` and the last notebook cell)** reads one line of input and prints, in order: preprocessing, syntax parsing, semantic analysis, information extraction, sentiment, translation into Tamil (`dest='ta'`), LSA summary, generated text for the prompt "Once upon a time", the answer to a fixed question about a short hard-coded context, a speech-recognition heading (the `recognize_speech()` call is commented out), and finally speaks "Welcome to NLP Project!". `spam_detection` and `recognize_speech` are imported but not called.

## 4. Why It Is Used

- Splitting an assistant into components makes each stage easy to understand, test and replace.
- Combining classical NLP (spaCy, NLTK, Naive Bayes, LSA) with pretrained transformer pipelines shows both rule/statistics-based and neural approaches in one project.
- The same components cover the main tasks a real assistant needs: understanding, extraction, classification, generation, answering and speech.

## 5. Mathematical Intuition

**Multinomial Naive Bayes (spam detection).** Each text is a vector of word counts $x_i$. The class with the highest posterior wins, assuming words are independent given the class:

```math
\hat{c} = \arg\max_{c} \; \log P(c) + \sum_{i} x_i \log P(w_i \mid c),
\qquad
P(w_i \mid c) = \frac{N_{c,i} + \alpha}{N_c + \alpha |V|}
```

$N_{c,i}$ is the count of word $i$ in class $c$, $|V|$ the vocabulary size and $\alpha = 1$ (scikit-learn's default Laplace smoothing).

**LSA summarization.** Build a term-by-sentence matrix $A$ and factorise it with singular value decomposition:

```math
A = U \Sigma V^{\top}
```

Each singular vector is a latent "topic"; sentences with large weights in $V$ for the strongest topics (large values in $\Sigma$) are selected as the summary.

**Lexicon-based sentiment (TextBlob).** Each known word has a polarity score in $[-1, 1]$; modifiers such as "very" or "not" scale or flip it, and the sentence polarity is an average of the scored words.

**Transformer models (generation and question answering)** use self-attention:

```math
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V
```

- *Text generation* predicts one token at a time: $P(w_1, \dots, w_T) = \prod_t P(w_t \mid w_{<t})$, until `max_length` is reached.
- *Extractive QA* predicts a start and an end position in the context and returns the span with the highest $P_{\text{start}}(i) \cdot P_{\text{end}}(j)$, $i \le j$.

**Preprocessing, POS tagging, dependency parsing and NER** follow the principles described in [Text_Preprocessing](../Text_Preprocessing/README.md#5-mathematical-intuition). Speech recognition is covered in [Voice_Assistants](../Voice_Assistants/README.md#5-mathematical-intuition).

## 6. Important Parameters

**General parameters**

| Parameter | Meaning |
|---|---|
| Model choice per task | Which pretrained model each pipeline uses |
| Generation length (`max_length`, `max_new_tokens`) | How long generated text can be |
| Summary length | Number of sentences returned |
| Source and target language | Translation direction |
| Naive Bayes smoothing `alpha` | Handles words not seen in a class |
| Vectorizer settings | Tokenization, lower-casing, n-grams |
| Speech settings | Timeouts, phrase limits, recognition language |

**Used in this code**

| Setting | Value |
|---|---|
| spaCy model | `en_core_web_sm` (loaded separately in each spaCy cell) |
| NLTK data | `punkt`, `stopwords`, `wordnet` |
| Punctuation regex | `r'[^\w\s]'` |
| Preprocessing order | stop-word removal, then stemming, then lemmatization of the stems |
| Spam model | `CountVectorizer()` and `MultinomialNB()` with default settings; 4 training sentences |
| Translation | `dest='fr'` by default; `dest='ta'` in `main.py` |
| Summarizer | sumy `LsaSummarizer`, `Tokenizer("english")`, 2 sentences |
| Text generation | `pipeline("text-generation")` without a model name (library default model), `max_length=30`, prompt "Once upon a time" in `main.py` |
| Question answering | `pipeline("question-answering")` without a model name (library default model) |
| Speech recognition | `r.listen(source)` with defaults, `recognize_google` |

## 7. Advantages

- Modular design: each capability is a separate function and can be developed or replaced on its own.
- Pretrained transformer pipelines give generation and question answering without any training.
- Classical components (spaCy, NLTK, Naive Bayes, LSA) are fast and easy to explain.
- Covers both text and voice input and output.

## 8. Disadvantages

- The integration layer is unfinished (the imported modules do not exist yet).
- Pipelines created without a model name depend on library defaults, which change between versions; the saved outputs show one environment where `pipeline("summarization")` was not available at all.
- Default transformer models are large downloads and run slowly on a CPU; some pipelines are rebuilt on every function call.
- The spam classifier is trained on four sentences, so it only demonstrates the method.
- Coreference resolution is simulated with a fixed print statement.
- Translation and speech recognition rely on unofficial or free web services and need internet access.

## 9. Real-World Applications

- Smart speakers and phone assistants
- Customer-support chatbots that classify, summarise and answer requests
- Email spam filtering
- Document summarisation and translation tools
- Question answering over help-centre or knowledge-base articles

## 10. Python Libraries Used

| Library | Imports | Used for |
|---|---|---|
| NLTK | `import nltk`, `from nltk.corpus import stopwords`, `from nltk.stem import PorterStemmer, WordNetLemmatizer` | Preprocessing |
| re (standard library) | `import re` | Punctuation removal |
| spaCy | `import spacy` | POS tags, dependencies, entities |
| TextBlob | `from textblob import TextBlob` | Sentiment polarity |
| scikit-learn | `from sklearn.feature_extraction.text import CountVectorizer`, `from sklearn.naive_bayes import MultinomialNB` | Spam detection |
| Transformers (PyTorch backend) | `from transformers import pipeline` | Text generation, question answering (and the unused summarization pipeline) |
| googletrans | `from googletrans import Translator` | Translation |
| sumy | `from sumy.summarizers.lsa import LsaSummarizer`, `from sumy.parsers.plaintext import PlaintextParser`, `from sumy.nlp.tokenizers import Tokenizer` | Extractive summarization |
| SpeechRecognition | `import speech_recognition as sr` | Microphone input |
| pyttsx3 | `import pyttsx3` | Text-to-speech |

## 11. Dataset Information

There is no dataset file.

- **User input**: one line of text typed at run time (`main.py`).
- **Spam detection**: four hard-coded sentences with labels (1 = spam, 0 = not spam) and one test sentence, `'win free prize'`.
- **Question answering**: a fixed question and a short hard-coded two-sentence context in `main.py`.
- **Pretrained resources**: spaCy `en_core_web_sm`; NLTK `punkt`, `stopwords` and `wordnet`; default transformer models for text generation, question answering and summarization, downloaded from the Hugging Face Hub on first use (several hundred MB or more each) and cached under `%USERPROFILE%\.cache\huggingface`.

## 12. How to Execute

From the repository root, after following "Getting started" in the [root README](../../README.md):

```powershell
cd 03_Natural_Language_Processing\Smart_Virtual_Assistant
python -m jupyter notebook nlp.ipynb
```

1. Run the component cells in order, from the preprocessing cell to the question-answering cell.
2. Skip the integration cell (it imports the missing modules) and the `!pip install preprocess` cell.
3. Call the functions from a new cell, for example:

```python
sample = "The city library opened a new reading room last week. Visitors said the space was bright and quiet."
print(preprocess_text(sample))
parse_syntax(sample)
analyze_semantics(sample)
extract_entities_relations(sample)
sentiment_analysis(sample)
spam_detection()
print(summarize_text(sample))
```

Notes:

- The first run downloads NLTK data, and the generation cell downloads a default summarization model; `generate_text` and `answer_question` download their default models on first call. All of this needs internet access.
- The generation cell creates `pipeline("summarization")` before it defines `translate_text`, `summarize_text` and `generate_text`. If that line raises `KeyError: "Unknown task summarization"` (as in the saved output of `nlp.ipynb`), the installed `transformers` release does not provide that task and those three functions are not defined.
- `translate_text` needs internet access. `requirements.txt` pins `googletrans==4.0.2`, whose `Translator` is used asynchronously in [Machine_Translation_App/app.py](../Machine_Translation_App/app.py) (`async with Translator(...)` and `await translator.translate(...)`); the notebook's synchronous call may need the same treatment with that version.
- `recognize_speech()` needs a microphone and PyAudio.
- `python main.py` stops at its first import until the eight modules exist in this folder.

## 13. Expected Output

There is no successful end-to-end run in the saved outputs. When the component functions are called individually they print or return the following:

| Function | Output |
|---|---|
| `preprocess_text` | A Python list of lower-cased, stemmed and lemmatized tokens without stop words or punctuation |
| `parse_syntax` | `Tokens & POS:` with one `token POS` line per token, then `Dependency Parsing:` with `token --> dep --> head` lines |
| `analyze_semantics` | `Named Entities:` with `text LABEL` lines, then the fixed "Coreference Resolution (Simulated)" line |
| `extract_entities_relations` | `Entities:` lines, then `Relation:` lines for the root token |
| `sentiment_analysis` | `Sentiment Polarity:` followed by a number between -1 and 1 |
| `spam_detection` | `Spam Prediction:` followed by `1` (spam) or `0` (not spam) for `'win free prize'` |
| `translate_text` | The translated string |
| `summarize_text` | Up to two sentences from the input |
| `generate_text` | The prompt followed by generated text, at most 30 tokens in total (`max_length=30` includes the prompt) |
| `answer_question` | A short answer span taken from the context |
| `speak_text` | Spoken audio, no printed output |

**Saved outputs in the notebooks** (from the author's runs, in two different environments):

| Notebook | Cell | Saved output |
|---|---|---|
| `nlp.ipynb` | Preprocessing | NLTK messages reporting `punkt`, `stopwords` and `wordnet` as already up to date |
| `nlp.ipynb` | Classifier | `ModuleNotFoundError: No module named 'textblob'` (TextBlob was not installed at the time; it is now listed in `requirements.txt`) |
| `nlp.ipynb` | Generation | `KeyError: "Unknown task summarization, ..."` raised by `pipeline("summarization")`. The list of available tasks printed with the error includes `text-generation` but not `summarization` or `question-answering` |
| `nlp.ipynb` | Integration | `ImportError: cannot import name 'preprocess_text' from 'preprocess'` (the unrelated PyPI package), followed by a kernel-crash notice |
| `nlp.ipynb` | `!pip install preprocess` | Installed the unrelated PyPI package `preprocess` 2.0.0 and `future` 1.0.0 |
| `nlp1.ipynb` | Generation | The summarization pipeline loaded with the library's default model and the message "Device set to use cpu" |
| `nlp1.ipynb` | Integration | `ModuleNotFoundError: No module named 'syntax_parse'` (raised on the second import line) |
| `nlp1.ipynb` | `!pip install syntax_parse` | `ERROR: No matching distribution found for syntax_parse` |

## Next steps

1. Save each component cell as a local module next to `main.py`: `preprocess.py`, `syntax_parse.py`, `semantic_analysis.py`, `info_extract.py`, `classifier.py`, `generation.py`, `speech_module.py`, `qa_module.py`.
2. Remove the `!pip install preprocess` and `!pip install syntax_parse` cells, since PyPI does not provide the project's own modules.
3. Pass explicit model names to the transformer pipelines and use a `transformers` release that provides the `summarization` and `question-answering` tasks, or drop the unused `summarizer` line.
4. Add the components listed in the notebook's planned layout that are not implemented yet: constituency parsing, word-sense disambiguation, real coreference resolution and topic classification.

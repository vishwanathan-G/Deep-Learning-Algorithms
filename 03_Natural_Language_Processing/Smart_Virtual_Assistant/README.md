# Smart Virtual Assistant using NLP (Partially Tested)

A modular NLP assistant that takes text or voice input, processes it, extracts information, classifies it and responds by translating, summarizing, generating text, answering a question and speaking. The component functions were written as cells of `nlp.ipynb`; on 22 November 2026 they were saved, unchanged, as the eight modules that `main.py` imports. Only some of the modules could be tested so far (see "Current status").

## Files in this folder

| File | Description |
|---|---|
| [`nlp.ipynb`](nlp.ipynb) | "Smart Virtual Assistant using NLP": a short description, the planned module layout (cell 2), one code cell per component (cells 4-11), the integration cell (cell 12, identical to `main.py`) and a markdown note (cell 13) that replaces the former `!pip install preprocess` cell. Saved outputs from the author's runs include errors (see section 13). |
| [`nlp1.ipynb`](nlp1.ipynb) | Earlier variant of the same component cells without the descriptive markdown. `parse_syntax` is defined twice, and the integration cell imports `text_preprocessing` instead of `preprocess` (there is no module of that name in this folder). Cell 10, which used to run `!pip install syntax_parse`, is now a markdown note. |
| [`main.py`](main.py) | Integration script, identical to cell 12 of `nlp.ipynb`. Imports the eight modules and calls each component on one line of user input. |
| [`preprocess.py`](preprocess.py), [`syntax_parse.py`](syntax_parse.py), [`semantic_analysis.py`](semantic_analysis.py), [`info_extract.py`](info_extract.py), [`classifier.py`](classifier.py), [`generation.py`](generation.py), [`speech_module.py`](speech_module.py), [`qa_module.py`](qa_module.py) | The eight component modules, saved verbatim from cells 4-11 of `nlp.ipynb`. Each file starts with a header naming its source cell. See the table in section 3. |

Cell numbers in this README and in the module headers count from 0 (the first cell of a notebook is cell 0).

## Current status

The project structure has been completed from the notebook's own code; the project is partially tested.

- **The modules were never missing code.** Cells 4-11 of `nlp.ipynb` contain every function that `main.py` imports, and cell 2 shows the planned file layout, but the cells had not been saved as `.py` files, so `main.py` and the integration cell could not import them. On 22 November 2026 they were saved verbatim as the eight modules listed in section 3, each with a header naming its source cell, and each module was checked to equal its cell. `main.py` is identical to cell 12.
- **One addition not from the notebook.** `translate_text` in `generation.py` has a clearly marked compatibility block. In googletrans 4.x, `Translator.translate` is asynchronous and returns a coroutine; the block runs the coroutine with `asyncio.run`. That cannot be done inside an event loop that is already running, such as Jupyter's, so calling `translate_text` from the notebook itself may still fail with googletrans 4.x.
- **`!pip install preprocess` replaced by a note.** Cell 13 of `nlp.ipynb` used to run `!pip install preprocess`. That installs an unrelated PyPI package (`preprocess` 2.0.0) which shadows the local module, and it caused the `ImportError` saved in the integration cell. The cell is now a markdown note. Anyone who ran it should remove the package with `python -m pip uninstall preprocess`. Cell 10 of `nlp1.ipynb` (`!pip install syntax_parse`; PyPI has no package of that name) is now a note too.
- **Tests on the test laptop (22 November 2026).** `preprocess.py` and `classifier.py` were tested (results in section 13). Not testable on that machine: `syntax_parse.py`, `semantic_analysis.py` and `info_extract.py` (spaCy's compiled dependencies are blocked there by Windows Smart App Control), `generation.py` and `qa_module.py` (they need `transformers` below 5 and PyTorch; the machine has `transformers` 5.16.1, which lacks the `summarization` and `question-answering` tasks, and no PyTorch) and `speech_module.py` (needs a microphone). `main.py` was therefore not run end to end.
- **Remaining limitations.**
  - Coreference resolution is simulated with a fixed print statement.
  - `main.py` ends by speaking "Welcome to NLP Project!" with `pyttsx3`.
  - `generation.py` loads a large summarization model when it is imported, although `summarize_text` uses sumy.
  - The saved outputs of the integration cell still show the old `ImportError` from before the modules existed.

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

**Modules** (saved from the component cells of `nlp.ipynb`; the file names follow the notebook's planned layout in cell 2):

| Module | Source cell in `nlp.ipynb` | Function(s) | What the code does |
|---|---|---|---|
| `preprocess.py` | 4 | `preprocess_text(text)` | Downloads the NLTK data `punkt`, `punkt_tab`, `stopwords` and `wordnet` when imported. Lower-cases, removes every character that is not a word character or whitespace (`re.sub(r'[^\w\s]', '', text)`), tokenizes with `nltk.word_tokenize`, drops English stop words, applies `PorterStemmer`, then applies `WordNetLemmatizer` to the stems, and returns the list |
| `syntax_parse.py` | 5 | `parse_syntax(text)` | spaCy `en_core_web_sm`: prints each token with its POS tag, then `token --> dependency --> head` for each token |
| `semantic_analysis.py` | 6 | `analyze_semantics(text)` | Prints spaCy named entities with labels, then a fixed example line under "Coreference Resolution (Simulated)" |
| `info_extract.py` | 7 | `extract_entities_relations(text)` | Prints entities, then a "Relation" line for each token whose dependency label is `ROOT` |
| `classifier.py` | 8 | `sentiment_analysis(text)`, `spam_detection()` | TextBlob sentiment polarity; a `CountVectorizer` + `MultinomialNB` model trained on four sample sentences (`'win money now'`, `'hello friend'`, `'buy now'`, `'how are you?'` with labels 1, 0, 1, 0) that predicts the label of `'win free prize'` |
| `generation.py` | 9 | `translate_text(text, dest='fr')`, `summarize_text(text)`, `generate_text(prompt)` | `translate` of a module-level googletrans `Translator()`, plus the compatibility block for googletrans 4.x (the only addition not from the notebook); sumy `LsaSummarizer` returning 2 sentences; `transformers.pipeline("text-generation")` with `max_length=30`. The module also creates `pipeline("summarization")` when imported, which `summarize_text` does not use |
| `speech_module.py` | 10 | `recognize_speech()`, `speak_text(text)` | SpeechRecognition microphone capture with `recognize_google`; `pyttsx3` speech output |
| `qa_module.py` | 11 | `answer_question(question, context)` | `transformers.pipeline("question-answering")`, returns the answer span |

**Integration (`main.py`, identical to cell 12 of `nlp.ipynb`)** reads one line of input and prints, in order: preprocessing, syntax parsing, semantic analysis, information extraction, sentiment, translation into Tamil (`dest='ta'`), LSA summary, generated text for the prompt "Once upon a time", the answer to a fixed question about a short hard-coded context, a speech-recognition heading (the `recognize_speech()` call is commented out), and finally speaks "Welcome to NLP Project!". `spam_detection` and `recognize_speech` are imported but not called.

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
| spaCy model | `en_core_web_sm` (loaded separately in `syntax_parse.py`, `semantic_analysis.py` and `info_extract.py`) |
| NLTK data | `punkt`, `punkt_tab`, `stopwords`, `wordnet` |
| Punctuation regex | `r'[^\w\s]'` |
| Preprocessing order | stop-word removal, then stemming, then lemmatization of the stems |
| Spam model | `CountVectorizer()` and `MultinomialNB()` with default settings; 4 training sentences |
| Translation | googletrans `Translator()` with default settings; `dest='fr'` by default, `dest='ta'` in `main.py`; with googletrans 4.x the returned coroutine is run with `asyncio.run` |
| Summarizer | sumy `LsaSummarizer`, `Tokenizer("english")`, 2 sentences |
| Text generation | `pipeline("text-generation")` without a model name (library default model), `max_length=30`, prompt "Once upon a time" in `main.py` |
| Question answering | `pipeline("question-answering")` without a model name (library default model) |
| Speech recognition | `r.listen(source)` with defaults, `recognize_google` |

## 7. Advantages

- Modular design: each capability is a separate module and can be developed or replaced on its own.
- Pretrained transformer pipelines give generation and question answering without any training.
- Classical components (spaCy, NLTK, Naive Bayes, LSA) are fast and easy to explain.
- Covers both text and voice input and output.

## 8. Disadvantages

- `main.py` has not been run end to end yet, and several modules could not be tested (see "Current status").
- Pipelines created without a model name depend on library defaults, which change between versions; the saved outputs show one environment where `pipeline("summarization")` was not available at all, and `transformers` 5.x provides neither the `summarization` nor the `question-answering` task.
- Default transformer models are large downloads and run slowly on a CPU. `generation.py` loads a summarization model when imported even though `summarize_text` uses sumy, and `generate_text` and `answer_question` rebuild their pipelines on every call.
- The spam classifier is trained on four sentences, so it only demonstrates the method.
- Coreference resolution is simulated with a fixed print statement.
- Translation and speech recognition rely on unofficial or free web services and need internet access. With googletrans 4.x the translation call is asynchronous; `generation.py` handles this with `asyncio.run`, which cannot be used inside Jupyter.

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
| Transformers (PyTorch backend; a release below 5) | `from transformers import pipeline` | Text generation, question answering (and the unused summarization pipeline) |
| googletrans | `from googletrans import Translator` | Translation |
| asyncio, inspect (standard library) | `import asyncio`, `import inspect` (`generation.py` only) | Running the coroutine that googletrans 4.x returns |
| sumy | `from sumy.summarizers.lsa import LsaSummarizer`, `from sumy.parsers.plaintext import PlaintextParser`, `from sumy.nlp.tokenizers import Tokenizer` | Extractive summarization |
| SpeechRecognition | `import speech_recognition as sr` | Microphone input |
| pyttsx3 | `import pyttsx3` | Text-to-speech |

## 11. Dataset Information

There is no dataset file.

- **User input**: one line of text typed at run time (`main.py`).
- **Spam detection**: four hard-coded sentences with labels (1 = spam, 0 = not spam) and one test sentence, `'win free prize'`.
- **Question answering**: a fixed question and a short hard-coded two-sentence context in `main.py`.
- **Pretrained resources**: spaCy `en_core_web_sm`; NLTK `punkt`, `punkt_tab`, `stopwords` and `wordnet`; default transformer models for text generation, question answering and summarization, downloaded from the Hugging Face Hub on first use (several hundred MB or more each) and cached under `%USERPROFILE%\.cache\huggingface`.

## 12. How to Execute

From the repository root, after following "Getting started" in the [root README](../../README.md).

**Dependencies** (all of these packages are in the root `requirements.txt`; the spaCy model and the NLTK data are downloaded separately as noted):

| Package | Needed by |
|---|---|
| `nltk`, with the data `punkt`, `punkt_tab`, `stopwords` and `wordnet` (downloaded by `preprocess.py` when it is imported) | `preprocess.py` |
| `spacy` and the model `en_core_web_sm` (`python -m spacy download en_core_web_sm`) | `syntax_parse.py`, `semantic_analysis.py`, `info_extract.py` |
| `textblob`, `scikit-learn` | `classifier.py` |
| `transformers` below 5, and `torch` | `generation.py` (`summarization` and `text-generation` pipelines) and `qa_module.py` (`question-answering` pipeline). The models download on first use. `generation.py` creates the summarization pipeline when it is imported, even though `summarize_text` uses sumy |
| `sumy` | `generation.py` (`summarize_text`) |
| `googletrans` 4.0.2 | `generation.py` (`translate_text`) |
| `SpeechRecognition` and `PyAudio` | `speech_module.py` (`recognize_speech`) |
| `pyttsx3` | `speech_module.py` (`speak_text`) |

If `!pip install preprocess` was ever run from the notebook, remove that unrelated package first:

```powershell
python -m pip uninstall preprocess
```

**Run the assistant** from this folder:

```powershell
cd 03_Natural_Language_Processing\Smart_Virtual_Assistant
python main.py
```

Type one line of text at the `Enter input:` prompt.

**Use the notebook** to run single components:

```powershell
python -m jupyter notebook nlp.ipynb
```

1. Run the component cells 4-11 in order.
2. Call the functions from a new cell, for example:

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

3. The integration cell (cell 12) does the same as `main.py`; prefer `python main.py`. Inside Jupyter, the notebook's own `translate_text` (cell 9, without the compatibility block) receives a coroutine from googletrans 4.x instead of a translation, and the `asyncio.run` in `generation.py` cannot be used in Jupyter's running event loop.

Notes:

- The first run downloads NLTK data. Importing `generation.py` (which `main.py` does at start-up), or running cell 9, creates the summarization pipeline and downloads its default model; `generate_text` and `answer_question` download their default models on first call. All of this needs internet access.
- If `pipeline("summarization")` raises `KeyError: "Unknown task summarization"` (as in the saved output of `nlp.ipynb`), the installed `transformers` release does not provide that task. `transformers` 5.x (for example 5.16.1) lacks both `summarization` and `question-answering`, so `requirements.txt` keeps `transformers` below 5.
- `translate_text` needs internet access.
- `recognize_speech()` needs a microphone and PyAudio; `main.py` does not call it (the line is commented out).
- `main.py` ends by speaking "Welcome to NLP Project!" through `pyttsx3`.

## 13. Expected Output

There is no successful end-to-end run in the saved outputs, and `main.py` has not been run end to end yet. When the component functions are called individually they print or return the following:

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

**Test results** (22 November 2026, test laptop with Windows 11):

| Module | Call | Result |
|---|---|---|
| `preprocess.py` | `preprocess_text('The students are running quickly to their classes!')` | `['student', 'run', 'quickli', 'class']` |
| `classifier.py` | `sentiment_analysis("I really love this wonderful project")` | Polarity `0.75` |
| `classifier.py` | `sentiment_analysis("This is a terrible, awful result")` | Polarity `-1.0` |
| `classifier.py` | `spam_detection()` (test sentence `'win free prize'`) | `1` (spam) |
| `syntax_parse.py`, `semantic_analysis.py`, `info_extract.py` | - | Not tested: spaCy's compiled dependencies are blocked by Windows Smart App Control on that machine |
| `generation.py`, `qa_module.py` | - | Not tested: they need `transformers` below 5 and PyTorch; the machine has `transformers` 5.16.1 and no PyTorch |
| `speech_module.py` | - | Not tested: needs a microphone |
| `main.py` | - | Not run end to end |

In addition, googletrans 4.x was confirmed to make `Translator.translate` asynchronous (it returns a coroutine), which is why `generation.py` has its compatibility block.

**Saved outputs in the notebooks** (from the author's runs, in two different environments). The Windows username in file paths was replaced with `<user>`, and the outputs of the two former `!pip install` cells went away when those cells became markdown notes.

| Notebook | Cell | Saved output |
|---|---|---|
| `nlp.ipynb` | Preprocessing (cell 4) | NLTK messages reporting `punkt`, `stopwords` and `wordnet` as already up to date (saved before the `punkt_tab` download was added to the cell) |
| `nlp.ipynb` | Classifier (cell 8) | `ModuleNotFoundError: No module named 'textblob'` (TextBlob was not installed at the time; it is now listed in `requirements.txt`) |
| `nlp.ipynb` | Generation (cell 9) | `KeyError: "Unknown task summarization, ..."` raised by `pipeline("summarization")`. The list of available tasks printed with the error includes `text-generation` but not `summarization` or `question-answering` |
| `nlp.ipynb` | Integration (cell 12) | `ImportError: cannot import name 'preprocess_text' from 'preprocess'`, raised by the unrelated PyPI package in `site-packages`, followed by a kernel-crash notice. This output dates from before the local modules existed |
| `nlp.ipynb` | Cell 13 | Now a markdown note. As a code cell (`!pip install preprocess`) it had installed the unrelated package `preprocess` 2.0.0 and `future` 1.0.0; that output is no longer in the notebook |
| `nlp1.ipynb` | Generation (cell 6) | The summarization pipeline loaded with the library's default model and the message "Device set to use cpu" |
| `nlp1.ipynb` | Integration (cell 9) | `ModuleNotFoundError: No module named 'syntax_parse'` (raised on the second import line, before `syntax_parse.py` existed) |
| `nlp1.ipynb` | Cell 10 | Now a markdown note. As a code cell (`!pip install syntax_parse`) it had ended with `ERROR: No matching distribution found for syntax_parse`; that output is no longer in the notebook |

## Next steps

1. Run `python main.py` end to end on a machine where spaCy loads, with `transformers` below 5, PyTorch, a microphone and speakers, so that `syntax_parse.py`, `semantic_analysis.py`, `info_extract.py`, `generation.py`, `qa_module.py` and `speech_module.py` are tested too.
2. Re-run the integration cell of `nlp.ipynb`, or clear its output, so that it no longer shows the `ImportError` from before the modules existed.
3. Pass explicit model names to the transformer pipelines, and remove the unused summarization pipeline that `generation.py` creates when imported (or create it only when needed).
4. Make `translate_text` usable from inside Jupyter with googletrans 4.x, where `asyncio.run` cannot be used.
5. Change the first import of the integration cell in `nlp1.ipynb` from `text_preprocessing` to `preprocess`, or treat `nlp1.ipynb` as an earlier draft.
6. Add the components listed in the notebook's planned layout that are not implemented yet: constituency parsing, word-sense disambiguation, real coreference resolution and topic classification.

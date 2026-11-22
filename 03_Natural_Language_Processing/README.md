# Natural Language Processing

Practice code for natural language processing (NLP): text preprocessing with spaCy and NLTK, voice-driven assistants, a modular virtual-assistant project (partially tested) and a Streamlit machine-translation app. This page gives an overview of the folder and documents the Machine Translation App; the other subfolders have their own README files.

## Files in this folder

| Folder | Files | Topic | Documentation |
|---|---|---|---|
| [`Text_Preprocessing/`](Text_Preprocessing/) | `code1.py`, `new.py` | Tokenization, stop-word removal, stemming, lemmatization, POS tagging, NER, dependency parsing; text-to-speech of the result | [README](Text_Preprocessing/README.md) |
| [`Voice_Assistants/`](Voice_Assistants/) | `with_voice.py`, `bot_with_voice.py` | Speech-to-text, NLTK pipeline on spoken input, spoken Wikipedia answers | [README](Voice_Assistants/README.md) |
| [`Smart_Virtual_Assistant/`](Smart_Virtual_Assistant/) | `nlp.ipynb`, `nlp1.ipynb`, `main.py` and eight modules saved from the notebook (`preprocess.py`, `syntax_parse.py`, `semantic_analysis.py`, `info_extract.py`, `classifier.py`, `generation.py`, `speech_module.py`, `qa_module.py`) | Modular assistant: preprocessing, parsing, NER, sentiment, spam detection, translation, summarization, generation, QA, speech (partially tested) | [README](Smart_Virtual_Assistant/README.md) |
| [`Machine_Translation_App/`](Machine_Translation_App/) | `app.py`, `requirements.txt`, `.streamlit/config.toml`, `.gitignore`, `README.md` | Streamlit translation app using `googletrans` | Sections 1-13 below and the [app's own README](Machine_Translation_App/README.md) |

## NLP overview

NLP is the field that lets computers read, analyse and produce human language. Most systems combine some of these stages, and each one appears somewhere in this folder:

| Stage | Purpose | Where it appears |
|---|---|---|
| Preprocessing | Split text into tokens and sentences; normalise with stop-word removal, stemming and lemmatization | `Text_Preprocessing`, `Voice_Assistants/with_voice.py`, `Smart_Virtual_Assistant` |
| Syntactic analysis | Label parts of speech and grammatical dependencies | `Text_Preprocessing/code1.py`, `Smart_Virtual_Assistant` |
| Semantic analysis and information extraction | Find named entities and relations | `Text_Preprocessing`, `Smart_Virtual_Assistant` |
| Classification | Assign labels such as sentiment or spam | `Smart_Virtual_Assistant` |
| Generation and transformation | Translate, summarise, generate or answer | `Machine_Translation_App`, `Smart_Virtual_Assistant` |
| Speech | Convert speech to text and text to speech | `Voice_Assistants`, `Text_Preprocessing/new.py`, `Smart_Virtual_Assistant` |

---

# Machine Translation App

`app.py` is a Streamlit web app titled "Lingua — Machine Translator". It sends the user's text to a web translation service through the unofficial `googletrans` package and shows the result. The app does not train or host a translation model: translation and language detection happen on the remote service, and the code in this repository handles the interface, input checks, the asynchronous request, the display and the download. The author's own [README for the app](Machine_Translation_App/README.md) covers setup inside that folder and an important note about the unofficial library.

## 1. Algorithm Name

Machine translation (neural machine translation performed by an external web service), with automatic source-language detection.

## 2. Definition

Machine translation (MT) automatically converts text in a source language into text with the same meaning in a target language. Current systems use neural machine translation (NMT): a neural network trained on large collections of sentence pairs learns the probability of a target sentence given a source sentence.

## 3. How It Works

**General idea (neural machine translation)**

1. The source text is split into subword tokens and mapped to vectors.
2. An **encoder** (today usually a Transformer) builds a context-aware representation of every source token.
3. A **decoder** generates the target sentence one token at a time. At every step it attends to the encoder's representations to decide which source words matter most.
4. A search procedure such as beam search keeps the most probable partial translations and returns the best complete one.
5. **Language identification** predicts the source language when it is not given.

**What `app.py` does**

1. Sets the page title, wide layout and an expanded sidebar, and injects CSS for fonts and colours. The light theme colours are in `.streamlit/config.toml`.
2. Initialises session state: empty source and translated text, source `Auto-detect`, target `English`.
3. Builds the language list from `googletrans.LANGUAGES`: names are title-cased and sorted alphabetically, with a name-to-code map.
4. Shows "Translate from" (with `Auto-detect` first) and "Translate to" select boxes and a swap button. Swapping while the source is `Auto-detect` shows the toast "Choose a source language first to swap languages."
5. Shows a source text area limited to 15,000 characters with a live character counter, and a read-only text area for the translation.
6. On **Translate text**, it checks that the text is not empty, not over the limit, and that source and target differ, then calls:

   ```python
   async with Translator(service_urls=["translate.googleapis.com"]) as translator:
       return await translator.translate(text, src=source_code, dest=target_code)
   ```

   through `asyncio.run`, using `"auto"` as the source code for automatic detection.
7. Stores the translated text and an information line ("Detected source: ... · Target: ..." or "Source: ... · Target: ...") and reruns the page to display them.
8. On failure it shows an error message asking the user to check the internet connection, plus the exception type and message.
9. **Clear** runs the `on_click` callback `clear_text()`, which sets the source text, the translation and the information line in session state to empty strings. Streamlit runs the callback before the page is redrawn, which is when it allows the key of an existing widget (`source_text`) to be reset; the swap button's `swap_languages` callback uses the same pattern. Earlier versions set `st.session_state.source_text` after the text area had been created, which raised `StreamlitAPIException`; on 22 November 2026 the reset was moved into `clear_text()` and the old block after the widgets, which ended with `st.rerun()`, was removed. **Download translation as .txt** saves the result as `translation.txt` (UTF-8).

## 4. Why It Is Used

- Makes text in an unfamiliar language readable within seconds.
- Automatic language detection removes the need to know the source language.
- Wrapping the service in a small web app shows how an NLP capability becomes a usable product: input validation, state handling, error messages and file export.

## 5. Mathematical Intuition

**Translation as conditional probability.** An NMT model with parameters $\theta$ factorises the probability of target sentence $y$ given source sentence $x$ token by token:

```math
P(y \mid x; \theta) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, x; \theta)
```

Training maximises $\sum \log P(y \mid x; \theta)$ over many sentence pairs. Decoding looks for $\hat{y} = \arg\max_y P(y \mid x)$, approximated with beam search.

**Attention** lets every decoder step weigh all source positions:

```math
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V
```

**Evaluation.** Translation quality is commonly measured with BLEU, which combines modified $n$-gram precisions $p_n$ against reference translations with a brevity penalty $BP$:

```math
\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)
```

## 6. Important Parameters

**General parameters**

| Parameter | Meaning |
|---|---|
| Source and target language | Translation direction; `auto` lets the system detect the source |
| Model or service | Which translation engine is used |
| Maximum input length | Longer texts must be split into chunks |
| Beam size and maximum output length | Search settings inside the model (not exposed by this service) |
| Domain or glossary | Terminology control, offered by some commercial APIs |

**Used in this code**

| Setting | Value in `app.py` |
|---|---|
| `MAX_CHARACTERS` | 15,000 (enforced by the text area's `max_chars` and checked again before translating) |
| Source language | `"auto"` when `Auto-detect` is selected, otherwise the chosen language code |
| Default languages | Source `Auto-detect`, target `English` |
| Translator | `Translator(service_urls=["translate.googleapis.com"])`, used as an async context manager |
| Language list | `googletrans.LANGUAGES`, title-cased and sorted with `str.casefold` |
| Text areas | Height 255 pixels; the translation area is read-only |
| Download | `translation.txt`, MIME type `text/plain; charset=utf-8` |
| Theme (`.streamlit/config.toml`) | `base = "light"`, primary colour `#3559e0`, sans-serif font |

## 7. Advantages

- No API key, billing or model download is needed.
- Supports the full language list exposed by `googletrans`, with automatic detection.
- Input checks and clear messages for empty text, the character limit, identical languages and failed requests.
- The translation can be saved as a UTF-8 text file.
- The whole app is a single Python script, run with Streamlit.

## 8. Disadvantages

- `googletrans` is unofficial and depends on a public web endpoint that can change or block requests without notice; it is not suited to production use.
- Requires an internet connection, and the text is sent to an external service.
- 15,000-character limit per request.
- No control over the model, terminology or tone.
- Machine translation can mistranslate idioms, ambiguous words and domain-specific terms.

## 9. Real-World Applications

- Translating websites, product descriptions and documentation
- Customer-support chat across languages
- Reading foreign-language news, research and emails
- Travel and live conversation apps
- First-draft translation followed by human post-editing

## 10. Python Libraries Used

| Library | Import in `app.py` | Used for |
|---|---|---|
| Streamlit | `import streamlit as st` | Web interface, session state, buttons, text areas, download button |
| googletrans | `from googletrans import LANGUAGES, Translator` | Language list and translation requests |
| asyncio (standard library) | `import asyncio` | Running the asynchronous translation call |
| `__future__` (standard library) | `from __future__ import annotations` | Postponed evaluation of type hints |

The app's own `requirements.txt` lists `streamlit>=1.40,<2.0` and `googletrans==4.0.2`; both are also in the root `requirements.txt`.

## 11. Dataset Information

There is no dataset and no training. The input is the text typed or pasted by the user. The list of languages comes from `googletrans.LANGUAGES`, and the translation is produced by the remote service.

## 12. How to Execute

From the repository root, after following "Getting started" in the [root README](../README.md):

```powershell
cd 03_Natural_Language_Processing\Machine_Translation_App
python -m streamlit run app.py
```

Run the command from inside the app folder so that Streamlit picks up `.streamlit/config.toml`. To install only the app's dependencies in a separate environment, follow the [app's README](Machine_Translation_App/README.md) or run `python -m pip install -r requirements.txt` inside this folder. An internet connection is required.

## 13. Expected Output

There are no saved outputs for the app; this section describes what it shows.

1. Streamlit prints a local URL in the terminal (by default `http://localhost:8501`) and opens the app in the browser.
2. The page shows a heading, a sidebar with a short note that the app uses the unofficial `googletrans` library and needs internet, the two language selectors with the swap button, the source and translation text areas, a character counter (`0 / 15,000 characters`), and the **Translate text** and **Clear** buttons.
3. After a successful translation, the translation area is filled, a line such as `Detected source: <language> · Target: <language>` appears under it, and a **Download translation as .txt** button is shown.
4. Messages for invalid input: "Enter some text first.", "Choose different source and target languages." or the character-limit error.
5. If the request fails: "Translation failed. Check your internet connection and try again..." with a "Technical detail" caption showing the exception.
6. **Clear** empties the source and translation boxes.

**Check on 22 November 2026** (Windows 11, Streamlit 1.57, Streamlit `AppTest`): the page renders without errors; after a sentence is typed the counter shows `19 / 15,000 characters`; clicking **Clear** raises no exception and empties both the source and the translation box. Translation itself was not exercised, because it needs internet access.

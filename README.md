# Deep Learning Algorithms

Practice implementations of deep-learning and natural language processing techniques in Python. The repository contains a convolutional neural network for image classification, a recurrent neural network for sentiment analysis, and a set of NLP scripts, notebooks and a Streamlit app covering text preprocessing, voice assistants, a modular virtual assistant (partially tested) and machine translation.

Every folder has a README that explains the technique (definition, how it works, mathematical intuition, parameters, advantages, disadvantages and applications) and then documents the code: libraries, data, how to run it and what it prints. The CNN and Smart Virtual Assistant notebooks keep the saved outputs of the author's runs; the RNN notebook's saved outputs come from a re-run on 22 November 2026.

## Table of contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Index](#index)
- [Saved results](#saved-results)
- [Getting started](#getting-started)
- [Hardware and internet notes](#hardware-and-internet-notes)
- [Changes on 22 November 2026](#changes-on-22-november-2026)
- [Verification](#verification)
- [Not yet covered](#not-yet-covered)
- [Author](#author)

## Overview

| Area | What is implemented |
|---|---|
| Computer vision | A Keras CNN (three convolution layers, two max-pooling layers, two dense layers) trained on CIFAR-10 |
| Sequence modelling | A Keras Embedding + SimpleRNN model trained on IMDB movie reviews |
| Natural language processing | spaCy and NLTK preprocessing pipelines, speech-driven NLTK processing, a spoken question-answering bot, a modular virtual assistant combining classical NLP with pretrained transformer pipelines, and a Streamlit translation app |

## Repository structure

```text
Deep-Learning-Algorithms/
├── .gitignore
├── README.md
├── requirements.txt
├── 01_Convolutional_Neural_Network/
│   ├── CNN.ipynb
│   └── README.md
├── 02_Recurrent_Neural_Network/
│   ├── README.md
│   └── RNN.ipynb
└── 03_Natural_Language_Processing/
    ├── README.md
    ├── Machine_Translation_App/
    │   ├── .streamlit/
    │   │   └── config.toml
    │   ├── .gitignore
    │   ├── app.py
    │   ├── README.md
    │   └── requirements.txt
    ├── Smart_Virtual_Assistant/
    │   ├── classifier.py
    │   ├── generation.py
    │   ├── info_extract.py
    │   ├── main.py
    │   ├── nlp.ipynb
    │   ├── nlp1.ipynb
    │   ├── preprocess.py
    │   ├── qa_module.py
    │   ├── README.md
    │   ├── semantic_analysis.py
    │   ├── speech_module.py
    │   └── syntax_parse.py
    ├── Text_Preprocessing/
    │   ├── code1.py
    │   ├── new.py
    │   └── README.md
    └── Voice_Assistants/
        ├── bot_with_voice.py
        ├── README.md
        └── with_voice.py
```

## Index

| Category | Technique | Folder | Files | Data source |
|---|---|---|---|---|
| Deep learning - computer vision | Convolutional Neural Network, 10-class image classification | [01_Convolutional_Neural_Network](01_Convolutional_Neural_Network/README.md) | `CNN.ipynb` | CIFAR-10, downloaded by `tf.keras.datasets.cifar10` (about 170 MB) |
| Deep learning - sequence models | Embedding + SimpleRNN, binary sentiment classification | [02_Recurrent_Neural_Network](02_Recurrent_Neural_Network/README.md) | `RNN.ipynb` | IMDB reviews, downloaded by `tf.keras.datasets.imdb` (about 17 MB) |
| NLP | Overview of the NLP folder; machine translation app | [03_Natural_Language_Processing](03_Natural_Language_Processing/README.md) | `Machine_Translation_App/app.py` | Text typed by the user; translation through the unofficial `googletrans` client (internet) |
| NLP | Text preprocessing: tokenization, stop-word removal, stemming, lemmatization, POS tagging, NER, dependency parsing | [Text_Preprocessing](03_Natural_Language_Processing/Text_Preprocessing/README.md) | `code1.py`, `new.py` | Text typed by the user; spaCy `en_core_web_sm`; NLTK data packages |
| NLP - speech | Speech-to-text, NLTK pipeline on spoken input, spoken question answering, text-to-speech | [Voice_Assistants](03_Natural_Language_Processing/Voice_Assistants/README.md) | `with_voice.py`, `bot_with_voice.py` | Microphone audio; Wikipedia summaries through the `wikipedia` library |
| NLP - assistant (partially tested) | Preprocessing, parsing, NER, sentiment, Naive Bayes spam detection, translation, LSA summarization, text generation, question answering, speech | [Smart_Virtual_Assistant](03_Natural_Language_Processing/Smart_Virtual_Assistant/README.md) | `nlp.ipynb`, `nlp1.ipynb`, `main.py`, `preprocess.py`, `syntax_parse.py`, `semantic_analysis.py`, `info_extract.py`, `classifier.py`, `generation.py`, `speech_module.py`, `qa_module.py` | Text typed by the user; four hard-coded spam examples; pretrained transformer models from the Hugging Face Hub |

The Smart Virtual Assistant's eight modules were saved from the component cells of `nlp.ipynb`, so `main.py` can now import them. Only some of the modules could be tested so far; its README lists the current status, the test results and the next steps.

## Saved results

Numbers below are saved outputs stored in the notebooks. They will differ on every run because no random seed is set.

| Notebook | Source of the saved output | Training | Test accuracy |
|---|---|---|---|
| `01_Convolutional_Neural_Network/CNN.ipynb` | The author's run | 5 epochs, batch size 32 (default), Adam | 0.6935 |
| `02_Recurrent_Neural_Network/RNN.ipynb` | Re-run on 22 November 2026 on the real IMDB dataset (the author's earlier numbers were replaced) | 5 epochs, batch size 64, Adam, 20% validation split | 0.8232 |

In the RNN re-run, 5 of the 6 sample test reviews (`x_test[5:11]`) are predicted correctly. Per-epoch metrics are in each folder's README.

## Getting started

The code was written for Python 3.10 (the notebooks' saved kernel is Python 3.10.11). The commands below are for Windows PowerShell. The repository is private for now, so cloning needs access to it.

```powershell
git clone https://github.com/vishwanathan-G/Deep-Learning-Algorithms.git
cd Deep-Learning-Algorithms

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If PowerShell blocks the activation script, call the environment's interpreter directly instead, for example `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

**Notebooks** (CNN, RNN, Smart Virtual Assistant):

```powershell
python -m jupyter notebook
```

Then open a notebook from the browser tab and run its cells in order.

**Scripts** (Text Preprocessing, Voice Assistants, and `main.py` of the Smart Virtual Assistant), from the script's folder (each example starts at the repository root):

```powershell
cd 03_Natural_Language_Processing\Text_Preprocessing
python code1.py
```

```powershell
cd 03_Natural_Language_Processing\Smart_Virtual_Assistant
python main.py
```

**Streamlit app** (Machine Translation), from the app's folder so that its `.streamlit/config.toml` is used:

```powershell
cd 03_Natural_Language_Processing\Machine_Translation_App
python -m streamlit run app.py
```

Each folder's README has the exact commands and notes for that folder.

## Hardware and internet notes

- **CPU is enough.** On native Windows, TensorFlow 2.11 and later do not use the GPU (the saved output of `CNN.ipynb` contains this warning). In the saved logs, CNN epochs took about 23-39 seconds on the author's machine; the RNN re-run on 22 November 2026 took about 9 minutes for the whole notebook on a CPU, with epochs of 32-225 seconds.
- **Memory.** After normalisation the CIFAR-10 images are `float64` arrays of about 1.5 GB (training and test together), so several GB of free RAM are advisable for the CNN notebook.
- **First-run downloads (internet required):**
  - CIFAR-10 (about 170 MB) and IMDB (about 17 MB), cached in `%USERPROFILE%\.keras\datasets`
  - spaCy `en_core_web_sm` (installed with `python -m spacy download en_core_web_sm`)
  - NLTK data packages, downloaded by the scripts and notebooks into `%APPDATA%\nltk_data`
  - Default transformer models used by the Smart Virtual Assistant (several hundred MB or more each), cached in `%USERPROFILE%\.cache\huggingface`
  - PyTorch and TensorFlow themselves are large installs
- **Internet at run time:** speech recognition through `recognize_google` (Google Web Speech API), translation through `googletrans`, and answers through the `wikipedia` library.
- **Microphone:** needed for `Voice_Assistants` and for `recognize_speech()` in the Smart Virtual Assistant; `PyAudio` provides microphone access.
- **Speakers:** `pyttsx3` speaks offline through the operating system's speech engine (SAPI5 on Windows).

## Changes on 22 November 2026

Changes made after a code-review round:

- **RNN labels.** The prediction cell of `RNN.ipynb` predicted `x_test[5:11]` but printed `y_test[i]` (the labels of reviews 0-5) as "Actual". It now prints `y_test[5 + i]`, the label of the same review. The notebook was re-run top to bottom on the real IMDB dataset and all saved outputs were regenerated: test accuracy 0.8232, 5 of 6 sample predictions correct.
- **Clear button.** In `Machine_Translation_App/app.py` the **Clear** button set `st.session_state.source_text` after the text area had been created, which raised `StreamlitAPIException`. The reset now runs in an `on_click` callback, `clear_text()` (the same pattern as `swap_languages`), and the old block with `st.rerun()` after the widgets was removed.
- **NLTK `punkt_tab`.** NLTK 3.10.3 raises a `LookupError` for `punkt_tab` when only `punkt` is downloaded. `nltk.download('punkt_tab')` was added after the `punkt` download in `Text_Preprocessing/new.py`, `Voice_Assistants/with_voice.py`, cell 4 of `nlp.ipynb` (and therefore `preprocess.py`) and cell 0 of `nlp1.ipynb`.
- **Smart Virtual Assistant modules.** The eight modules imported by `main.py` were never missing code: cells 4-11 of `nlp.ipynb` contain every function, but they had not been saved as `.py` files. They are now saved verbatim, each with a header naming its source cell; `main.py` is identical to cell 12. The only addition is a marked compatibility block in `generation.py` for the asynchronous `translate` of googletrans 4.x. The `!pip install preprocess` cell (it installed an unrelated PyPI package that shadows the local module) and the `!pip install syntax_parse` cell are now markdown notes.
- **Privacy.** The Windows username in file paths inside saved outputs was replaced with `<user>` in `CNN.ipynb`, `nlp.ipynb` and `nlp1.ipynb`. A scan of all files found no other Windows usernames, local drive paths, phone numbers or credentials; the only email address is the author's contact address at the end of this README.

## Verification

State of the checks on 22 November 2026 (Windows 11, Python 3.10.11, TensorFlow 2.21.0 / Keras 3.12.4). The saved outputs of `CNN.ipynb`, `nlp.ipynb` and `nlp1.ipynb` are the author's own runs (apart from the `<user>` replacement and the two `!pip install` cells that became notes); the saved outputs of `RNN.ipynb` come from the re-run below.

| Item | Check | Result |
|---|---|---|
| `CNN.ipynb` | All cells run top to bottom, with small random stand-in arrays in place of the CIFAR-10 download (earlier check; the notebook's code has not changed since) | Passed (6 of 6 cells). This confirms the model builds, trains, evaluates and predicts; accuracy on the real dataset was not re-measured. |
| `RNN.ipynb` | Re-run top to bottom on the real IMDB dataset from the Keras cache, on the CPU (about 9 minutes) | Passed (6 of 6 cells); all saved outputs regenerated. Test accuracy 0.8232; validation loss lowest after epoch 2 (0.4200). The actual labels of the six sample reviews `x_test[5:11]` (Positive, Positive, Negative, Negative, Positive, Positive) were verified independently through `imdb.load_data`; 5 of 6 predictions are correct (Review 4: predicted Positive, actual Negative). |
| `Machine_Translation_App/app.py` | Streamlit `AppTest` (Streamlit 1.57): load the page, type a sentence, click **Clear** | Page renders without errors; after typing, the counter shows `19 / 15,000 characters`; **Clear** raises no exception and empties both the source and the translation box. Translation itself was not exercised (it needs internet). |
| `Text_Preprocessing/new.py`, `Voice_Assistants/with_voice.py` | Run end to end with the real NLTK on a clean NLTK data folder; microphone (`speech_recognition`) and speaker (`pyttsx3`) replaced by test stand-ins; `with_voice.py` received the sample sentence as if it had been recognised | Passed: tokenization, stop-word removal, stemming, lemmatization, POS tags and named entities were produced (`['Barack', 'Obama', 'Hawaii', 'United States']` for "Barack Obama was born in Hawaii. He was the president of the United States.") and the final text reached the speaker stand-in. The real microphone, speaker and Google recogniser were not tested. |
| `Smart_Virtual_Assistant` | Each module compared with its notebook cell; modules called on the test laptop | Partially tested. Every module equals its source cell. `preprocess_text('The students are running quickly to their classes!')` returned `['student', 'run', 'quickli', 'class']`; `sentiment_analysis` gave 0.75 for "I really love this wonderful project" and -1.0 for "This is a terrible, awful result"; `spam_detection()` returned 1 (spam) for `'win free prize'`. Not testable there: `syntax_parse`, `semantic_analysis`, `info_extract` (spaCy blocked by Windows Smart App Control), `generation`, `qa_module` (they need `transformers` below 5 and PyTorch; the machine has `transformers` 5.16.1 and no PyTorch) and `speech_module` (microphone). `main.py` was not run end to end. |
| `Text_Preprocessing/code1.py` | - | Not run: spaCy could not load on the test machine because Windows Smart App Control blocked one of its compiled dependencies. |
| `Voice_Assistants/bot_with_voice.py` | - | Not run: it needs a microphone and the `wikipedia` library. |

## Not yet covered

Standalone ANN (multilayer perceptron), LSTM, GRU, autoencoder and transfer-learning implementations are not in this repository yet.

## Author

**Vishwanathan G**

- GitHub: <https://github.com/vishwanathan-G>
- Email: <g.vishwanathan29@gmail.com>

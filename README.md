# Deep Learning Algorithms

Practice implementations of deep-learning and natural language processing techniques in Python. The repository contains a convolutional neural network for image classification, a recurrent neural network for sentiment analysis, and a set of NLP scripts, notebooks and a Streamlit app covering text preprocessing, voice assistants, a modular virtual assistant (work in progress) and machine translation.

Every folder has a README that explains the technique (definition, how it works, mathematical intuition, parameters, advantages, disadvantages and applications) and then documents the code: libraries, data, how to run it and what it prints. The notebooks keep the saved outputs of the author's runs.

## Table of contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Index](#index)
- [Saved results](#saved-results)
- [Getting started](#getting-started)
- [Hardware and internet notes](#hardware-and-internet-notes)
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
    │   ├── main.py
    │   ├── nlp.ipynb
    │   ├── nlp1.ipynb
    │   └── README.md
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
| NLP - assistant (work in progress) | Preprocessing, parsing, NER, sentiment, Naive Bayes spam detection, translation, LSA summarization, text generation, question answering, speech | [Smart_Virtual_Assistant](03_Natural_Language_Processing/Smart_Virtual_Assistant/README.md) | `nlp.ipynb`, `nlp1.ipynb`, `main.py` | Text typed by the user; four hard-coded spam examples; pretrained transformer models from the Hugging Face Hub |

The Smart Virtual Assistant's integration step imports local modules that are not in the repository yet; its README lists the current status and next steps.

## Saved results

Numbers below are saved output from the author's runs, stored in the notebooks. They will differ slightly on every run because no random seed is set.

| Notebook | Training | Test accuracy |
|---|---|---|
| `01_Convolutional_Neural_Network/CNN.ipynb` | 5 epochs, batch size 32 (default), Adam | 0.6935 |
| `02_Recurrent_Neural_Network/RNN.ipynb` | 5 epochs, batch size 64, Adam, 20% validation split | 0.8216 |

Per-epoch metrics are in each folder's README.

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

**Scripts** (Text Preprocessing, Voice Assistants), from the script's folder:

```powershell
cd 03_Natural_Language_Processing\Text_Preprocessing
python code1.py
```

**Streamlit app** (Machine Translation), from the app's folder so that its `.streamlit/config.toml` is used:

```powershell
cd 03_Natural_Language_Processing\Machine_Translation_App
python -m streamlit run app.py
```

Each folder's README has the exact commands and notes for that folder.

## Hardware and internet notes

- **CPU is enough.** On native Windows, TensorFlow 2.11 and later do not use the GPU (the saved notebook outputs contain this warning). In the saved logs, CNN epochs took about 23-39 seconds and RNN epochs about 21-24 seconds on the author's machine.
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

## Verification

Checks run on 22 November 2026 before this repository was published (Windows 11, Python 3.10.11, TensorFlow 2.21.0 / Keras 3.12.4). The notebooks' saved outputs are the author's own runs and were not changed.

| Item | Check | Result |
|---|---|---|
| All 3 notebooks and 6 scripts | Notebook JSON valid; every code cell and script parses | Passed |
| `CNN.ipynb` | All cells run top to bottom, with small random stand-in arrays in place of the CIFAR-10 download | Passed (6 of 6 cells). This confirms the model builds, trains, evaluates and predicts; accuracy on the real dataset was not re-measured. |
| `RNN.ipynb` | Same, with stand-in arrays in place of the IMDB download | Passed (6 of 6 cells); accuracy not re-measured |
| `Machine_Translation_App/app.py` | Page loaded with Streamlit's `AppTest` | Page renders without errors. Clicking **Clear** raises a `StreamlitAPIException` (see the NLP README). Translation itself was not exercised (it needs internet). |
| `Text_Preprocessing/code1.py` | Run with a sample sentence | Not run: spaCy could not load on the test machine because Windows Application Control blocked one of its compiled dependencies. Syntax is valid. |
| `new.py`, `with_voice.py`, `bot_with_voice.py` | Import check | Not run: they need `pyttsx3`, `SpeechRecognition` and `wikipedia` (not installed on the test machine), plus a microphone and internet. Syntax is valid. |
| `Smart_Virtual_Assistant` | Import check | Not runnable as is: it imports 8 local modules that are not in the repository (see its README). With `transformers` 5.x the `summarization` and `question-answering` pipeline tasks do not exist, so `requirements.txt` keeps `transformers` below 5. |

## Not yet covered

Standalone ANN (multilayer perceptron), LSTM, GRU, autoencoder and transfer-learning implementations are not in this repository yet.

## Author

**Vishwanathan G**

- GitHub: <https://github.com/vishwanathan-G>
- Email: <g.vishwanathan29@gmail.com>

# Lingua — Machine Translator

A Streamlit classroom demonstration for translating text with the Python `googletrans` package. It includes automatic source-language detection, source/target selection, swap, character counting, readable output, and a translation download.

## Important note for students

This project uses `googletrans`, an **unofficial** Python library that communicates with Google Translate's web interface. It is not Google's official Cloud Translation API and is not associated with Google. It needs an internet connection, has a documented 15,000-character limit per text, and can stop working if the web service changes or temporarily blocks requests. It does not require a Google Cloud API key or billing setup.

For a reliable production application, use the official Google Cloud Translation API, which requires a Google Cloud project and billing.

## Run on Windows

Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, you can skip activation and run the venv Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Features

- Source language selection or automatic detection
- Target language selection from the package language list
- Swap source and target
- Character count and input limit
- Download translation as UTF-8 text
- No Google Cloud key or billing configuration

## Project files

- `app.py` — Streamlit application and translation flow
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — light theme configuration
- `.gitignore` — keeps virtual environments and secrets out of version control

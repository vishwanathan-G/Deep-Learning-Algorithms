# generation.py - Translation, Summarization, Text Gen
# Saved from cell 9 of nlp.ipynb, as planned in the notebook's project layout,
# so that main.py can import it. The code below is the notebook cell's code.
# One addition: the marked googletrans compatibility block in translate_text().

import asyncio
import inspect

#!pip install sumy

from transformers import pipeline
from googletrans import Translator
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

summarizer = pipeline("summarization")
translator = Translator()

def translate_text(text, dest='fr'):
    result = translator.translate(text, dest=dest)
    # --- compatibility fix (not in the notebook) ---------------------------------
    # googletrans 4.x makes translate() asynchronous, so it returns a coroutine.
    # Run it to get the translation object; older 3.x versions return it directly.
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    # -----------------------------------------------------------------------------
    return result.text

def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summary = LsaSummarizer()(parser.document, 2)
    return " ".join(str(sentence) for sentence in summary)

def generate_text(prompt):
    generator = pipeline("text-generation")
    result = generator(prompt, max_length=30)
    return result[0]['generated_text']

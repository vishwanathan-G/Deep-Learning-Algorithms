# Voice Assistants - Speech Input, NLP Processing and Spoken Answers

Two microphone-driven scripts: one turns speech into text and runs an NLTK preprocessing pipeline on it, the other answers spoken questions with a short Wikipedia summary read aloud.

## Files in this folder

| File | Description |
|---|---|
| [`with_voice.py`](with_voice.py) | Records one spoken phrase, converts it to text with `recognize_google`, runs the NLTK pipeline (counts, tokens, stop-word removal, stemming, lemmatization, POS tagging, named entities) and reads the lemmatized words aloud with `pyttsx3`. |
| [`bot_with_voice.py`](bot_with_voice.py) | Question-answering loop: listens for a question, looks it up with `wikipedia.summary(question, sentences=2)` and speaks the answer. Saying "goodbye" ends the loop. |

## 1. Algorithm Name

Speech-driven NLP:

- Automatic speech recognition (ASR, speech-to-text)
- Text preprocessing: tokenization, stop-word removal, stemming, lemmatization, POS tagging, named-entity chunking (`with_voice.py`)
- Retrieval-based question answering from an online encyclopedia (`bot_with_voice.py`)
- Text-to-speech (TTS) synthesis

## 2. Definition

A voice assistant connects three stages: **speech recognition** converts audio into text, a **language-processing** stage analyses the text or finds an answer, and **speech synthesis** turns the response back into audio. In `bot_with_voice.py` the answer comes from **retrieval**: the recognised question is used as a search query and the opening sentences of the best-matching article are returned.

## 3. How It Works

**`with_voice.py`**

1. Downloads the NLTK data it needs (`punkt`, `stopwords`, `averaged_perceptron_tagger_eng`, `maxent_ne_chunker`, `maxent_ne_chunker_tab`, `words`, `wordnet`).
2. Creates the stop-word set, `PorterStemmer`, `WordNetLemmatizer` and a `pyttsx3` engine.
3. Prints a "Speak now..." prompt, opens the default microphone, calibrates for background noise with `adjust_for_ambient_noise`, and records with `listen(source, timeout=10, phrase_time_limit=30)`.
4. Sends the audio to `recognize_google`. If recognition fails, it prints "Could not recognize your voice:" with the error and exits.
5. Runs the same NLTK steps as `Text_Preprocessing/new.py`: word and sentence counts, tokens, stop-word and punctuation removal, Porter stems, WordNet lemmas, POS tags of the filtered tokens and named-entity phrases from `ne_chunk`.
6. Joins the lemmatized words and speaks them.

**`bot_with_voice.py`**

1. Creates a `Recognizer` and a `pyttsx3` engine. The helper `speak(text)` prints the text and says it.
2. Speaks the greeting "Hi! I am a random question answering bot. Ask me anything!"
3. In a loop: calibrates for background noise, records with `listen(source, timeout=10, phrase_time_limit=20)` and converts the audio with `recognize_google`.
4. If the question contains "goodbye", says goodbye and stops.
5. Otherwise calls `wikipedia.summary(question, sentences=2)`:
   - a `DisambiguationError` gives "Your question is too broad. Try asking about: " followed by the first suggested page,
   - a `PageError` gives "Sorry, I couldn't find information on that topic."
6. Speaks the answer.
7. Handles `WaitTimeoutError` ("I heard nothing. Please ask again."), `UnknownValueError` ("Sorry, I couldn't understand your voice.") and any other exception ("An error occurred.", with the error printed).

## 4. Why It Is Used

- Speech is the most natural interface for many users and the only practical one in hands-busy or eyes-busy situations.
- Combining ASR with NLP shows how spoken input can feed the same text pipeline used for typed input.
- Retrieval from an existing knowledge source gives factual answers without training a model.

## 5. Mathematical Intuition

**Speech recognition** searches for the word sequence $W$ that is most probable given the audio features $X$. Rewriting the conditional probability (the denominator $P(X)$ is the same for every candidate) splits this into an acoustic model and a language model:

```math
\hat{W} = \arg\max_{W} P(W \mid X) = \arg\max_{W} P(X \mid W)\, P(W)
```

Modern recognisers implement this with neural networks trained end to end, but the decomposition still explains the two kinds of knowledge involved: how words sound and which word sequences are likely.

**Noise calibration and phrase detection.** `adjust_for_ambient_noise` measures the root-mean-square (RMS) energy of about one second of background audio and sets an energy threshold from it. While listening, audio whose energy rises above the threshold is treated as speech, and a phrase ends after a pause.

```math
\text{RMS} = \sqrt{\frac{1}{N}\sum_{n=1}^{N} s_n^2}
```

**Retrieval-based answering.** The question string is used as a search query; the library takes the best-matching page and returns its first sentences. No scoring of candidate answers is done in the script, so the quality of the answer depends on how well the spoken question works as a search query.

**Preprocessing** (`with_voice.py`) follows the same principles as [Text_Preprocessing](../Text_Preprocessing/README.md#5-mathematical-intuition): rule-based stemming, dictionary-based lemmatization, perceptron POS tagging and maximum-entropy entity chunking.

## 6. Important Parameters

**General parameters**

| Parameter | Meaning |
|---|---|
| `timeout` | Seconds to wait for speech to start before giving up |
| `phrase_time_limit` | Maximum length of one recorded phrase |
| `energy_threshold` / ambient-noise duration | Loudness level treated as speech |
| Recognition language | Language and accent of the recogniser |
| Answer length | Number of sentences returned from the knowledge source |
| TTS voice, rate and volume | How the answer sounds |

**Used in this code**

| Setting | `with_voice.py` | `bot_with_voice.py` |
|---|---|---|
| Noise calibration | `adjust_for_ambient_noise(source)` (default duration) | Same, before every question |
| `listen` timeout | 10 s | 10 s |
| `phrase_time_limit` | 30 s | 20 s |
| Recogniser | `recognize_google(audio)` with its default language (US English) | Same |
| Text processing | NLTK pipeline (see section 3) | None; the whole question is the search query |
| Answer source | - | `wikipedia.summary(question, sentences=2)` |
| Exit | After one phrase | When the question contains "goodbye" |
| TTS | `pyttsx3` default voice | `pyttsx3` default voice |

## 7. Advantages

- Hands-free interaction.
- `pyttsx3` speaks offline through the operating system's speech engine.
- The recogniser needs no API key and no model training.
- The bot handles the common failure cases (silence, unclear speech, ambiguous or missing pages) with spoken messages.
- Small, readable scripts that show the full speech-to-speech loop.

## 8. Disadvantages

- Recognition and answering need an internet connection, and audio is sent to an external web service.
- Accuracy drops with background noise, accents, microphone quality and domain-specific words.
- The bot does not interpret the question; question phrasing ("what is", "who was") becomes part of the search query.
- Only the first disambiguation option is suggested, which may not be what the user meant.
- No conversation memory: each question is handled independently.

## 9. Real-World Applications

- Smart speakers and phone voice assistants
- In-car voice control
- Accessibility tools for users with visual or motor impairments
- Voice search and dictation
- Spoken FAQ and help-desk bots

## 10. Python Libraries Used

| Library | Imports | Used for |
|---|---|---|
| SpeechRecognition | `import speech_recognition as sr` | Microphone capture, noise calibration, `recognize_google` |
| PyAudio | (used internally by `sr.Microphone`) | Microphone access |
| pyttsx3 | `import pyttsx3` | Offline text-to-speech |
| NLTK | `import nltk`, `from nltk.corpus import stopwords`, `from nltk.stem import PorterStemmer, WordNetLemmatizer`, `from nltk import pos_tag, word_tokenize, ne_chunk`, `from nltk.tree import Tree` (`with_voice.py`) | Preprocessing pipeline |
| wikipedia | `import wikipedia` (`bot_with_voice.py`) | Article search and summaries |
| string (standard library) | `import string` (`with_voice.py`) | Punctuation list |

## 11. Dataset Information

There is no dataset file. Input is live microphone audio. At run time:

- speech is transcribed by the free Google Web Speech API that `recognize_google` calls,
- `bot_with_voice.py` fetches article summaries from Wikipedia through the `wikipedia` library,
- `with_voice.py` downloads NLTK data packages to `%APPDATA%\nltk_data` on first use.

## 12. How to Execute

From the repository root, after following "Getting started" in the [root README](../../README.md):

```powershell
cd 03_Natural_Language_Processing\Voice_Assistants
python with_voice.py
python bot_with_voice.py
```

Requirements and notes:

- A working microphone and the `PyAudio` package (included in `requirements.txt`).
- Internet access for speech recognition, Wikipedia and the first NLTK download.
- Speakers or headphones to hear the `pyttsx3` output.
- Start speaking within 10 seconds of the prompt.
- Say "goodbye" to stop `bot_with_voice.py` (or press `Ctrl+C`).
- NLTK 3.9 and later load `punkt_tab` for tokenization, while `with_voice.py` downloads `punkt`. If a `LookupError` mentions `punkt_tab`, run `python -c "import nltk; nltk.download('punkt_tab')"` once.

## 13. Expected Output

There are no saved outputs for these scripts; this section describes what they print.

**`with_voice.py`**

1. `[nltk_data]` download or "already up-to-date" messages.
2. A "Speak now..." prompt, then "You said:" followed by the recognised text (or "Could not recognize your voice:" with the error, after which the script exits).
3. The sections `---- Text Statistics ----`, `---- Tokens ----`, `---- After Stopword Removal ----`, `---- Stemming ----`, `---- Lemmatization ----`, `---- Part of Speech (POS) Tagging ----` and `---- Named Entities ----`, each followed by a count or a Python list.
4. `Reading Processed Text...` while the lemmatized words are spoken, then `Process Complete.`

**`bot_with_voice.py`**

1. The spoken and printed greeting.
2. For each round: a "Listening... (Say your question)" prompt, `You asked: <recognised text>`, and a two-sentence Wikipedia summary that is printed and spoken. One of the fallback messages from section 3 is spoken instead when nothing is heard, the speech is not understood, the topic is ambiguous or no page is found.
3. "Goodbye! See you next time." when the question contains "goodbye".

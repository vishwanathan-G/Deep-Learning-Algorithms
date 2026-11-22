# speech_module.py - Speech-to-text and Text-to-speech
# Saved from cell 10 of nlp.ipynb, as planned in the notebook's project layout,
# so that main.py can import it. The code below is the notebook cell's code.

import speech_recognition as sr
import pyttsx3

def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak:")
        audio = r.listen(source)
        text = r.recognize_google(audio)
        return text

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

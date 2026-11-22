import wikipedia
import speech_recognition as sr
import pyttsx3

# Initialize speech engines
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def speak(text):
    print("🤖", text)
    engine.say(text)
    engine.runAndWait()

# Voice Bot Loop
speak("Hi! I am a random question answering bot. Ask me anything!")

while True:
    try:
        with sr.Microphone() as source:
            print("\n🎙️ Listening... (Say your question)")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)

        question = recognizer.recognize_google(audio)
        print("You asked:", question)

        if 'goodbye' in question.lower():
            speak("Goodbye! See you next time.")
            break

        # Fetch answer from Wikipedia
        try:
            answer = wikipedia.summary(question, sentences=2)
        except wikipedia.exceptions.DisambiguationError as e:
            answer = f"Your question is too broad. Try asking about: {e.options[0]}"
        except wikipedia.exceptions.PageError:
            answer = "Sorry, I couldn't find information on that topic."

        speak(answer)

    except sr.WaitTimeoutError:
        speak("I heard nothing. Please ask again.")
    except sr.UnknownValueError:
        speak("Sorry, I couldn't understand your voice.")
    except Exception as e:
        speak("An error occurred.")
        print(e)



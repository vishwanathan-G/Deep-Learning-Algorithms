from preprocess import preprocess_text
from syntax_parse import parse_syntax
from semantic_analysis import analyze_semantics
from info_extract import extract_entities_relations
from classifier import sentiment_analysis, spam_detection
from generation import translate_text, summarize_text, generate_text
from speech_module import recognize_speech, speak_text
from qa_module import answer_question

if __name__ == "__main__":
    text = input("Enter input: ")

    print("\n--- Preprocessing ---")
    print(preprocess_text(text))

    print("\n--- Syntax Parsing ---")
    parse_syntax(text)

    print("\n--- Semantic Analysis ---")
    analyze_semantics(text)

    print("\n--- Info Extraction ---")
    extract_entities_relations(text)

    print("\n--- Sentiment Analysis ---")
    sentiment_analysis(text)

    print("\n--- Translation ---")
    print(translate_text(text, dest='ta'))  # Tamil

    print("\n--- Summarization ---")
    print(summarize_text(text))

    print("\n--- Text Generation ---")
    print(generate_text("Once upon a time"))

    print("\n--- QA ---")
    context = "John is a scientist who works at NASA. He discovered a new planet."
    print(answer_question("Who is John?", context))

    print("\n--- Speech Recognition ---")
    # print(recognize_speech())  # Uncomment if using mic

    print("\n--- TTS ---")
    speak_text("Welcome to NLP Project!")

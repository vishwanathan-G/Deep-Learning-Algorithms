import spacy
from nltk.stem import PorterStemmer

# Load spaCy's English model.
# It provides POS tags, lemmas, dependency labels, and named entities.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise SystemExit(
        "The spaCy English model is missing.\n"
        "Install it by running:\n"
        "python -m spacy download en_core_web_sm"
    )

stemmer = PorterStemmer()


def analyze_sentence(text: str) -> None:
    """Analyze an English sentence and print common NLP outputs."""
    doc = nlp(text)

    print("\n" + "=" * 80)
    print("ORIGINAL TEXT")
    print("=" * 80)
    print(text)

    # Tokens, lemmas, stopword status, and grammatical information
    print("\n" + "=" * 80)
    print("TOKEN-LEVEL ANALYSIS")
    print("=" * 80)
    print(
        f"{'TOKEN':<18} {'STEM':<18} {'LEMMA':<18} "
        f"{'POS':<10} {'TAG':<8} {'STOPWORD':<10} {'DEPENDENCY'}"
    )
    print("-" * 100)

    for token in doc:
        if token.is_space:
            continue

        print(
            f"{token.text:<18} "
            f"{stemmer.stem(token.text.lower()):<18} "
            f"{token.lemma_:<18} "
            f"{token.pos_:<10} "
            f"{token.tag_:<8} "
            f"{str(token.is_stop):<10} "
            f"{token.dep_}"
        )

    # Keep only non-stopword words; discard punctuation and spaces too
    filtered_tokens = [
        token.text
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
    ]

    # Lemmatized version after removing stopwords and punctuation
    lemmatized_filtered = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
    ]

    # Stemmed version after removing stopwords and punctuation
    stemmed_filtered = [
        stemmer.stem(token.text.lower())
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
    ]

    print("\n" + "=" * 80)
    print("STOPWORD REMOVAL")
    print("=" * 80)
    print("Words kept:", filtered_tokens)

    print("\n" + "=" * 80)
    print("LEMMATIZED TEXT (stopwords removed)")
    print("=" * 80)
    print(" ".join(lemmatized_filtered))

    print("\n" + "=" * 80)
    print("STEMMED TEXT (stopwords removed)")
    print("=" * 80)
    print(" ".join(stemmed_filtered))

    # Named entities such as people, places, organizations, dates, and money
    print("\n" + "=" * 80)
    print("NAMED ENTITIES")
    print("=" * 80)

    if doc.ents:
        for entity in doc.ents:
            print(f"{entity.text:<30} {entity.label_}")
    else:
        print("No named entities found.")

    # Sentence boundaries
    print("\n" + "=" * 80)
    print("SENTENCES")
    print("=" * 80)

    for sentence in doc.sents:
        print("-", sentence.text)


if __name__ == "__main__":
    print("English NLP Analyzer")
    print("Type a sentence and press Enter.")
    print("Type 'quit' to exit.")

    while True:
        user_text = input("\nEnter text: ").strip()

        if user_text.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if not user_text:
            print("Please enter a sentence.")
            continue

        analyze_sentence(user_text)
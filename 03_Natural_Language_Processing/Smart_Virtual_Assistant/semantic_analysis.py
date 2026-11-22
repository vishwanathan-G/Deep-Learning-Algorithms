# semantic_analysis.py - NER, WSD, Coref
# Saved from cell 6 of nlp.ipynb, as planned in the notebook's project layout,
# so that main.py can import it. The code below is the notebook cell's code.

import spacy

nlp = spacy.load("en_core_web_sm")

def analyze_semantics(text):
    doc = nlp(text)
    print("Named Entities:")
    for ent in doc.ents:
        print(ent.text, ent.label_)

    print("\nCoreference Resolution (Simulated):")
    print("He is John. → 'He' refers to 'John'")

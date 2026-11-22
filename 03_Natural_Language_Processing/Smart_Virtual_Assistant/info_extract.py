# info_extract.py - Entity, Relation extraction
# Saved from cell 7 of nlp.ipynb, as planned in the notebook's project layout,
# so that main.py can import it. The code below is the notebook cell's code.

import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities_relations(text):
    doc = nlp(text)
    print("Entities:")
    for ent in doc.ents:
        print(ent.text, ent.label_)

    print("\nRelations:")
    for token in doc:
        if token.dep_ == "ROOT":
            print(f"Relation: {token.head.text} → {token.text}")

# syntax_parse.py - POS, Dependency, Constituency
# Saved from cell 5 of nlp.ipynb, as planned in the notebook's project layout,
# so that main.py can import it. The code below is the notebook cell's code.

#!python -m spacy download en_core_web_sm
import spacy

nlp = spacy.load("en_core_web_sm")

def parse_syntax(text):
    doc = nlp(text)
    print("Tokens & POS:")
    for token in doc:
        print(token.text, token.pos_)

    print("\nDependency Parsing:")
    for token in doc:
        print(f"{token.text} --> {token.dep_} --> {token.head.text}")

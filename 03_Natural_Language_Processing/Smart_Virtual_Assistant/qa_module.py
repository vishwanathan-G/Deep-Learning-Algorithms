# qa_module.py - Question Answering
# Saved from cell 11 of nlp.ipynb, as planned in the notebook's project layout,
# so that main.py can import it. The code below is the notebook cell's code.

from transformers import pipeline

def answer_question(question, context):
    qa_pipeline = pipeline("question-answering")
    result = qa_pipeline(question=question, context=context)
    return result['answer']

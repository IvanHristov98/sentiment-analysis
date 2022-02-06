from typing import List

import spacy
from spacy.lang.en import English

from sentiment.airline import Comment


def train(corpus: List[Comment]):
    nlp = spacy.load("en_core_web_sm")

    for comment in corpus:
        tokenize(nlp, comment.text)


def tokenize(nlp: English, text: str) -> List[str]:
    doc = nlp(text)
    tokens = []

    for token in doc:
        tokens.append(token.text)

    return tokens

from typing import List

import spacy
from spacy.lang.en import English
from spacy.tokens import Token

from sentiment.airline import Comment


def train(corpus: List[Comment]):
    nlp = spacy.load("en_core_web_sm")
    cnt = 0

    for comment in corpus:
        tokens = tokenize(nlp, comment.text)
        tokens = filter_stop_words(tokens)

        cnt += 1


def tokenize(nlp: English, text: str) -> List[Token]:
    doc = nlp(text)
    tokens = []

    for token in doc:
        tokens.append(token)

    return tokens


def filter_stop_words(tokens: List[Token]) -> List[Token]:
    return list(filter(lambda token: not token.is_stop, tokens))

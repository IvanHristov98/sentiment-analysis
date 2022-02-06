import re
from typing import List, Pattern

import spacy
from spacy.lang.en import English
from spacy.tokens import Token

from sentiment.airline import Comment


_TAG_REGEX = re.compile(r"^@.*$")
_URL_REGEX = re.compile(r"^https?:\/\/\S+|www\.\S+$")
_PUNCT_REGEX = re.compile(r"[^\w\s]")
_NEWLINE_REGEX = re.compile(r"\n")


def train(corpus: List[Comment]):
    nlp = spacy.load("en_core_web_sm")
    cnt = 0

    for comment in corpus:
        tokens = tokenize(nlp, comment.text)
        if cnt == 483:
            print(tokens)
        tokens = filter_stop_words(tokens)
        tokens = lemmatize(tokens)
        tokens = filter_junk(tokens)

        if cnt == 483:
            print(tokens)

        cnt += 1


def tokenize(nlp: English, text: str) -> List[Token]:
    doc = nlp(text)
    tokens = []

    for token in doc:
        tokens.append(token)

    return tokens


def filter_stop_words(tokens: List[Token]) -> List[Token]:
    return list(filter(lambda token: not token.is_stop, tokens))


def lemmatize(tokens: List[Token]) -> List[str]:
    return list(map(lambda token: token.lemma_, tokens))


def filter_junk(tokens: List[str]) -> List[str]:
    tokens = filter_by_regex(_TAG_REGEX, tokens)
    tokens = filter_by_regex(_URL_REGEX, tokens)
    tokens = filter_by_regex(_PUNCT_REGEX, tokens)
    tokens = filter_by_regex(_NEWLINE_REGEX, tokens)

    return tokens


def filter_by_regex(regex: Pattern, tokens: List[str]) -> List[str]:
    return list(filter(lambda token: not regex.search(token), tokens))

import re
from typing import List, Pattern

import spacy
from spacy.lang.en import English
from spacy.tokens import Token
import nltk

from sentiment.airline import Comment


_TAG_REGEX = re.compile(r"^@.*$")
_URL_REGEX = re.compile(r"^https?:\/\/\S+|www\.\S+$")
_PUNCT_REGEX = re.compile(r"[^\w\s]")
_NEWLINE_REGEX = re.compile(r"\n")
_EMPTY_REGEX = re.compile(r"^\s*$")

_WORD_FEATURE_COUNT = 2000
_BIGRAM_FEAUTRE_COUNT = 1000
_TRIGRAM_FEATURE_COUNT = 500


def train_classifier(corpus: List[Comment]):
    _features(corpus)


def _features(corpus: List[Comment]) -> List[str]:
    nlp = spacy.load("en_core_web_sm")

    words = []
    bigrams = []
    trigrams = []

    for comment in corpus:
        tokens = _tokenize(nlp, comment.text)
        tokens = _filter_stop_words(tokens)
        tokens = _lemmatize(tokens)
        tokens = _filter_junk(tokens)
        tokens = _lowercase(tokens)

        words += tokens
        bigrams += list(nltk.bigrams(tokens))
        trigrams += list(nltk.trigrams(tokens))

    features = _relevant_features(words, _WORD_FEATURE_COUNT)
    features += _relevant_features(bigrams, _BIGRAM_FEAUTRE_COUNT)
    features += _relevant_features(trigrams, _TRIGRAM_FEATURE_COUNT)

    return features


def _tokenize(nlp: English, text: str) -> List[Token]:
    doc = nlp(text)
    tokens = []

    for token in doc:
        tokens.append(token)

    return tokens


def _filter_stop_words(tokens: List[Token]) -> List[Token]:
    return list(filter(lambda token: not token.is_stop, tokens))


def _lemmatize(tokens: List[Token]) -> List[str]:
    return list(map(lambda token: token.lemma_, tokens))


def _filter_junk(tokens: List[str]) -> List[str]:
    tokens = _filter_by_regex(_TAG_REGEX, tokens)
    tokens = _filter_by_regex(_URL_REGEX, tokens)
    tokens = _filter_by_regex(_PUNCT_REGEX, tokens)
    tokens = _filter_by_regex(_NEWLINE_REGEX, tokens)
    tokens = _filter_by_regex(_EMPTY_REGEX, tokens)

    return tokens


def _filter_by_regex(regex: Pattern, tokens: List[str]) -> List[str]:
    return list(filter(lambda token: not regex.search(token), tokens))


def _lowercase(tokens: List[str]) -> List[str]:
    return list(map(lambda token: token.lower(), tokens))


def _relevant_features(ngrams: List[str], relevancy_threshold: int) -> List[str]:
    ngrams_by_freq = nltk.FreqDist(ngrams)
    return list(ngrams_by_freq)[:relevancy_threshold]

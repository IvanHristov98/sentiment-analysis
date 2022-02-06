import re
from typing import List, Pattern, Tuple, Dict

from spacy.lang.en import English
from spacy.tokens import Token
import nltk

from sentiment.airline import Comment


_TAG_REGEX = re.compile(r"^@.*$")
_URL_REGEX = re.compile(r"^https?:\/\/\S+|www\.\S+$")
_PUNCT_REGEX = re.compile(r"[^\w\s]")
_NEWLINE_REGEX = re.compile(r"\n")
_EMPTY_REGEX = re.compile(r"^\s*$")
_CONTAINS_DIGIT_REGEX = re.compile(r"\d")

_WORD_FEATURE_COUNT = 500
_BIGRAM_FEAUTRE_COUNT = 250
_TRIGRAM_FEATURE_COUNT = 100


def classifier(nlp: English, corpus: List[Comment]) -> Tuple[List[str], nltk.NaiveBayesClassifier]:
    print("Getting features...")
    features = extract_features(nlp, corpus)
    print("Labeling the features...")
    labeled_feature_sets = extract_labeled_feature_sets(nlp, corpus, features)

    print("Training the NBC...")
    return (features, nltk.NaiveBayesClassifier.train(labeled_feature_sets))


def extract_labeled_feature_sets(
    nlp: English, corpus: List[Comment], features: List[str]
) -> List[Tuple[Dict[str, bool], str]]:
    labeled_feature_sets = []

    for comment in corpus:
        comment_features = _feature_set(nlp, comment, features)
        labeled_feature_sets.append((comment_features, comment.sentiment.value))

    return labeled_feature_sets


def _feature_set(nlp: English, comment: Comment, features: List[str]) -> Dict[str, bool]:
    comment_features = set(extract_features(nlp, [comment]))
    feature_set = {}

    for feature in features:
        feature_set[feature] = feature in comment_features

    return feature_set


def extract_features(nlp: English, corpus: List[Comment]) -> List[str]:
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
        bigrams += _tuples_to_strings(list(nltk.bigrams(tokens)))
        trigrams += _tuples_to_strings(list(nltk.trigrams(tokens)))

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
    tokens = _filter_by_regex(_CONTAINS_DIGIT_REGEX, tokens)

    return tokens


def _filter_by_regex(regex: Pattern, tokens: List[str]) -> List[str]:
    return list(filter(lambda token: not regex.search(token), tokens))


def _lowercase(tokens: List[str]) -> List[str]:
    return list(map(lambda token: token.lower(), tokens))


def _tuples_to_strings(tuples: List[Tuple]) -> List[str]:
    return list(map(" ".join, tuples))


def _relevant_features(ngrams: List[str], relevancy_threshold: int) -> List[str]:
    ngrams_by_freq = nltk.FreqDist(ngrams)
    return list(ngrams_by_freq)[:relevancy_threshold]

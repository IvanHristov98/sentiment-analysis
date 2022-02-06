from typing import List
import math
import random

from spacy.lang.en import English
import nltk

from sentiment.airline import Comment
from sentiment import analysis

_K = 10


def run_fold_cross_validations(nlp: English, corpus: List[Comment]) -> None:
    random.shuffle(corpus)
    test_corpus_size = math.floor(len(corpus) / _K)
    test_corpus_offset = 0
    overall_accuracy = 0

    for k in range(_K):
        print(f"Running cross-fold validation {k}...")

        test_corpus = corpus[test_corpus_offset : test_corpus_offset + test_corpus_size]
        train_corpus = corpus[:test_corpus_offset] + corpus[test_corpus_offset + test_corpus_size :]

        accuracy = _run_cross_fold_validation(nlp, train_corpus, test_corpus)
        print(f"Accuracy of run {k} was {accuracy}.")

        overall_accuracy += accuracy / float(_K)
        test_corpus_offset += test_corpus_size

    print(f"Overall accuracy is {overall_accuracy}.")


def _run_cross_fold_validation(nlp: English, train_corpus: List[Comment], test_corpus: List[Comment]) -> None:
    features, classifier = analysis.classifier(nlp, train_corpus)
    test_feature_sets = analysis.extract_labeled_feature_sets(nlp, test_corpus, features)

    return nltk.classify.accuracy(classifier, test_feature_sets)

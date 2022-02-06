#!/usr/bin/env python3

import spacy

import sentiment.parse as parser
import sentiment.cmd.config as cfg
from sentiment import analysis
from sentiment.airline import Comment, Sentiment


def main():
    airline_comments_path = cfg.airline_comments_path()
    corpus = parser.parse_airline_comments(airline_comments_path)

    nlp = spacy.load("en_core_web_sm")

    _, classifier = analysis.classifier(nlp, corpus)
    classifier.show_most_informative_features()

    loop(nlp, classifier)


def loop(nlp, classifier):
    while True:
        comment_text = input("Enter a comment> ")
        comment = Comment(Sentiment.UNKNOWN, comment_text)

        features = analysis.extract_features(nlp, [comment])
        feature_set = analysis.extract_feature_set(nlp, comment, features)

        c = classifier.classify(feature_set)
        print(f"Comment classified as {c}.")


if __name__ == "__main__":
    main()

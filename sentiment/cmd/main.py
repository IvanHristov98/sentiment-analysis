#!/usr/bin/env python3

import spacy

import sentiment.parse as parser
import sentiment.cmd.config as cfg
from sentiment import analysis


def main():
    airline_comments_path = cfg.airline_comments_path()
    corpus = parser.parse_airline_comments(airline_comments_path)

    nlp = spacy.load("en_core_web_sm")

    classifier = analysis.classifier(nlp, corpus)
    classifier.show_most_informative_features()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import spacy

import sentiment.parse as parser
import sentiment.cmd.config as cfg
from sentiment import analysis
import sentiment.test as test


def main():
    airline_comments_path = cfg.airline_comments_path()
    corpus = parser.parse_airline_comments(airline_comments_path)

    nlp = spacy.load("en_core_web_sm")
    test.run_fold_cross_validations(nlp, corpus)


if __name__ == "__main__":
    main()

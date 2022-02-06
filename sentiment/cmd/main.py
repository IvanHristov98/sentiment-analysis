#!/usr/bin/env python3

import sentiment.parse as parser
import sentiment.cmd.config as cfg
from sentiment import analysis


def main():
    airline_comments_path = cfg.airline_comments_path()
    corpus = parser.parse_airline_comments(airline_comments_path)

    analysis.train_classifier(corpus)


if __name__ == "__main__":
    main()

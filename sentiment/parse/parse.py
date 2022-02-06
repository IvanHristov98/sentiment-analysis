import csv
from pathlib import Path
from typing import List

from sentiment.airline import Comment, sentiment_from_str


_SENTIMENT_IDX = 1
_TEXT_IDX = 10
_HEAD_ROW_IDX = 0


def parse_airline_comments(comments_src: Path) -> List[Comment]:
    comments = []

    with open(comments_src, encoding="utf-8") as stream:
        reader = csv.reader(stream, delimiter=",")
        row_cnt = 0

        for row in reader:
            if row_cnt == _HEAD_ROW_IDX:
                row_cnt += 1
                continue

            comment = _parse_airline_comment(row)
            comments.append(comment)

            row_cnt += 1

    return comments


def _parse_airline_comment(raw_comment: List[str]) -> Comment:
    sentiment = sentiment_from_str(raw_comment[_SENTIMENT_IDX])
    text = raw_comment[_TEXT_IDX]

    return Comment(sentiment, text)

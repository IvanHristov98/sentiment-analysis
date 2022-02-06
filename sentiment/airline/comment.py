from enum import Enum
from typing import NamedTuple


class Sentiment(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


def sentiment_from_str(raw_sentiment: str) -> Sentiment:
    switch = {
        Sentiment.POSITIVE.value: Sentiment.POSITIVE,
        Sentiment.NEUTRAL.value: Sentiment.NEUTRAL,
        Sentiment.NEGATIVE.value: Sentiment.NEGATIVE,
        Sentiment.UNKNOWN.value: Sentiment.UNKNOWN,
    }

    return switch[raw_sentiment]


class Comment(NamedTuple):
    sentiment: Sentiment
    text: str

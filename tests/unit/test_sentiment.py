import pytest
from src.features.sentiment import score_sentiment, average_sentiment


def test_positive_text():
    assert score_sentiment("great service, very happy") > 0


def test_negative_text():
    assert score_sentiment("terrible service, very unhappy") < 0


def test_empty_string():
    assert score_sentiment("") == 0.0


def test_whitespace_string():
    assert score_sentiment("   ") == 0.0


def test_average_sentiment_empty():
    assert average_sentiment([]) == 0.0


def test_average_sentiment_mixed():
    result = average_sentiment(["great service", "terrible service"])
    assert isinstance(result, float)


def test_average_sentiment_all_empty():
    assert average_sentiment(["", "", ""]) == 0.0

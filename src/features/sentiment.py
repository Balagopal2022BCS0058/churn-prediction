from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_sentiment(text: str) -> float:
    """Return compound sentiment score in [-1.0, 1.0]. Returns 0.0 for empty text."""
    if not text or not text.strip():
        return 0.0
    return _analyzer.polarity_scores(text)["compound"]


def average_sentiment(texts: list[str]) -> float:
    """Return average sentiment polarity across a list of texts."""
    if not texts:
        return 0.0
    scores = [score_sentiment(t) for t in texts]
    return sum(scores) / len(scores)

"""Unit tests for SentimentAnalyzer with lightweight fakes."""

from src.ai_engine.sentiment_analyzer import SentimentAnalyzer
from src.core.aspect_manager import AspectManager


class FakeClassifier:
    """Simple test double for classifier dependency."""

    def classify_sentiment(self, text, aspect_name):
        if aspect_name == "battery":
            return {"sentiment": "positive", "confidence": 0.90}
        if aspect_name == "screen":
            return {"sentiment": "negative", "confidence": 0.60}
        return {"sentiment": "neutral", "confidence": 0.80}

    def classify(self, review_text, labels, multi_label=False, hypothesis_template="{}"):
        return {"labels": labels, "scores": [0.8, 0.1, 0.1]}

    def get_model_info(self):
        return {"name": "fake-model"}


def test_analyze_review_applies_tier_thresholds():
    manager = AspectManager("electronics")
    analyzer = SentimentAnalyzer(
        category_name="electronics",
        aspect_manager=manager,
        classifier=FakeClassifier(),
        use_keyword_filter=True,
        min_confidence_tier1=0.55,
        min_confidence_tier2=0.70,
        use_negation_handling=False,
    )

    results = analyzer.analyze_review("Battery is awesome but screen is dim.")
    names = {item["aspect"] for item in results}

    assert "battery" in names
    assert "screen" not in names


def test_analyze_review_uses_cache_for_same_text():
    manager = AspectManager("electronics")
    analyzer = SentimentAnalyzer(
        category_name="electronics",
        aspect_manager=manager,
        classifier=FakeClassifier(),
        use_negation_handling=False,
    )

    text = "Battery performance is great."
    first = analyzer.analyze_review(text)
    second = analyzer.analyze_review(text)

    assert first == second
    assert analyzer.get_cache_size() == 1

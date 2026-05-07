"""Golden cases for sentiment calibration (_resolve_sentiment) and neutral gating."""

import pytest

from src.ai_engine.models.zero_shot import ZeroShotClassifier
from src.ai_engine.sentiment_analyzer import SentimentAnalyzer
from src.core.aspect_manager import AspectManager


@pytest.mark.parametrize(
    "scores,expected",
    [
        # Strong polarity
        ({"positive": 0.85, "negative": 0.10, "neutral": 0.05}, "positive"),
        ({"positive": 0.10, "negative": 0.85, "neutral": 0.05}, "negative"),
        # Weak top score -> neutral
        ({"positive": 0.40, "negative": 0.35, "neutral": 0.25}, "neutral"),
        # Ambiguous top-2 (small margin) -> neutral
        ({"positive": 0.55, "negative": 0.52, "neutral": 0.03}, "neutral"),
        # Neutral is top and strong enough
        ({"positive": 0.28, "negative": 0.30, "neutral": 0.42}, "neutral"),
        # Near tie three-way
        ({"positive": 0.34, "negative": 0.33, "neutral": 0.33}, "neutral"),
        # Polar separation without neutral rescue
        ({"positive": 0.72, "negative": 0.18, "neutral": 0.10}, "positive"),
        # Close polar + meaningful neutral -> neutral
        ({"positive": 0.48, "negative": 0.42, "neutral": 0.26}, "neutral"),
    ],
)
def test_resolve_sentiment_golden(scores, expected):
    assert ZeroShotClassifier._resolve_sentiment(scores) == expected


class _ClassifierNeutral5025:
    """Returns score map that resolves to neutral via weak top score."""

    def classify_sentiment(self, text, aspect_name):
        return {
            "sentiment": "neutral",
            "confidence": 0.50,
            "all_scores": {"positive": 0.30, "negative": 0.25, "neutral": 0.50},
        }


class _ClassifierAlwaysPositive:
    """Test double that lets neutral heuristic override polarity."""

    def classify_sentiment(self, text, aspect_name):
        return {
            "sentiment": "positive",
            "confidence": 0.70,
            "all_scores": {"positive": 0.70, "negative": 0.20, "neutral": 0.10},
        }


def test_neutral_passes_lower_min_confidence_gate():
    manager = AspectManager("electronics")
    analyzer = SentimentAnalyzer(
        category_name="electronics",
        aspect_manager=manager,
        classifier=_ClassifierNeutral5025(),
        use_keyword_filter=False,
        min_confidence_tier1=0.90,
        min_confidence_tier2=0.90,
        min_confidence_neutral=0.40,
        use_negation_handling=False,
    )
    results = analyzer.analyze_review(
        "Battery life is okay; screen is fine; nothing stands out."
    )
    assert results, "Expected at least one aspect when keyword filter is off"
    assert any(r["sentiment"] == "neutral" for r in results)


def test_neutral_dropped_when_below_min_confidence_neutral():
    manager = AspectManager("electronics")
    analyzer = SentimentAnalyzer(
        category_name="electronics",
        aspect_manager=manager,
        classifier=_ClassifierNeutral5025(),
        use_keyword_filter=False,
        min_confidence_tier1=0.90,
        min_confidence_tier2=0.90,
        min_confidence_neutral=0.90,
        use_negation_handling=False,
    )
    results = analyzer.analyze_review(
        "Battery life is okay; screen is fine; nothing stands out."
    )
    neutral_hits = [r for r in results if r["sentiment"] == "neutral"]
    assert not neutral_hits


def test_neutral_heuristic_overrides_mild_language():
    manager = AspectManager("electronics")
    analyzer = SentimentAnalyzer(
        category_name="electronics",
        aspect_manager=manager,
        classifier=_ClassifierAlwaysPositive(),
        use_keyword_filter=True,
        min_confidence_neutral=0.35,
        use_negation_handling=False,
    )
    # Contains neutral cue ("decent", "not great") and explicit aspect keyword.
    results = analyzer.analyze_review("Battery life is decent, not great.")
    assert results
    assert any(r["sentiment"] == "neutral" for r in results)


def test_value_positive_phrase_stays_positive():
    manager = AspectManager("electronics")
    analyzer = SentimentAnalyzer(
        category_name="electronics",
        aspect_manager=manager,
        classifier=_ClassifierAlwaysPositive(),
        use_keyword_filter=True,
        min_confidence_neutral=0.35,
        use_negation_handling=False,
    )
    results = analyzer.analyze_review("This is a great price and a good value.")
    assert results
    value_rows = [r for r in results if r["aspect"] == "value"]
    assert value_rows
    assert value_rows[0]["sentiment"] == "positive"


def test_sound_not_triggered_by_volume_keys_phrase():
    manager = AspectManager("electronics")
    analyzer = SentimentAnalyzer(
        category_name="electronics",
        aspect_manager=manager,
        classifier=_ClassifierAlwaysPositive(),
        use_keyword_filter=True,
        use_negation_handling=False,
    )
    results = analyzer.analyze_review(
        "The volume keys are intermittent and the special buttons stop working."
    )
    assert not any(r["aspect"] == "sound" for r in results)


def test_value_negative_phrase_forces_negative():
    manager = AspectManager("electronics")
    analyzer = SentimentAnalyzer(
        category_name="electronics",
        aspect_manager=manager,
        classifier=_ClassifierAlwaysPositive(),
        use_keyword_filter=True,
        use_negation_handling=False,
    )
    results = analyzer.analyze_review("This is overpriced and not worth the money.")
    value_rows = [r for r in results if r["aspect"] == "value"]
    assert value_rows
    assert value_rows[0]["sentiment"] == "negative"

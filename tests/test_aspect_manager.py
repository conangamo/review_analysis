"""Unit tests for AspectManager."""

from src.core.aspect_manager import AspectManager


def test_detect_aspects_by_keywords_returns_expected_aspects():
    manager = AspectManager("electronics")
    text = "Battery life is great and the screen is very bright."

    detected = manager.detect_aspects_by_keywords(text)
    names = {aspect["name"] for aspect in detected}

    assert "battery" in names
    assert "screen" in names


def test_get_aspects_for_analysis_strict_mode_uses_keyword_matches_only():
    manager = AspectManager("electronics")
    text = "The battery is excellent."

    aspects = manager.get_aspects_for_analysis(
        text,
        include_tier1_always=False,
        strict_keyword_matching=True,
    )
    names = {aspect["name"] for aspect in aspects}

    assert "battery" in names
    assert "performance" not in names


def test_should_display_aspect_obeys_tier_and_min_mentions():
    manager = AspectManager("electronics")

    assert manager.should_display_aspect("battery", mention_count=1) is True
    assert manager.should_display_aspect("screen", mention_count=2) is False
    assert manager.should_display_aspect("screen", mention_count=10) is True

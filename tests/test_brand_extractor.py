"""Unit tests for BrandExtractor."""

from src.core.brand_extractor import BrandExtractor


def test_extract_brand_prefers_details_brand_and_normalizes():
    extractor = BrandExtractor("electronics")
    product = {"details": {"Brand": "apple"}, "title": "Something else"}

    assert extractor.extract_brand(product) == "Apple"


def test_extract_brand_falls_back_to_store():
    extractor = BrandExtractor("electronics")
    product = {"details": {}, "store": "anker", "title": "Generic title"}

    assert extractor.extract_brand(product) == "Anker"


def test_extract_brand_from_title_skips_blacklist_word():
    extractor = BrandExtractor("electronics")
    product = {"details": {}, "title": "Wireless Sony Noise Cancelling Headphones"}

    assert extractor.extract_brand(product) == "Sony"

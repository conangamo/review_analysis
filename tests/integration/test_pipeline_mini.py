"""Mini integration test for parse -> load -> analyze flow."""

import gzip
import json
from pathlib import Path

from src.ai_engine.batch_processor import BatchProcessor
from src.core.brand_extractor import BrandExtractor
from src.core.config_loader import ConfigLoader
from src.data_processing.loader import DataLoader
from src.data_processing.parser import DataParser
from src.database.db_manager import DatabaseManager
from src.database.models import AspectSentiment, Review


class FakeAnalyzer:
    """Deterministic analyzer for integration testing."""

    def analyze_review(self, review_text, review_id=None):
        text = (review_text or "").lower()
        results = []
        if "battery" in text:
            results.append(
                {"aspect": "battery", "tier": 1, "sentiment": "positive", "confidence": 0.92}
            )
        if "screen" in text:
            results.append(
                {"aspect": "screen", "tier": 2, "sentiment": "negative", "confidence": 0.81}
            )
        return results

    def get_cache_size(self):
        return 0


def _write_jsonl_gz(path: Path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_mini_pipeline_parse_load_analyze(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = raw_dir / "meta_Electronics.jsonl.gz"
    reviews_path = raw_dir / "Electronics.jsonl.gz"
    db_path = tmp_path / "mini_pipeline.db"

    _write_jsonl_gz(
        metadata_path,
        [
            {
                "parent_asin": "P1",
                "title": "Sony Headphones",
                "average_rating": 4.4,
                "rating_number": 20,
                "details": {"Brand": "Sony"},
            },
            {
                "parent_asin": "P2",
                "title": "Anker Speaker",
                "average_rating": 4.1,
                "rating_number": 18,
                "details": {"Brand": "Anker"},
            },
        ],
    )

    _write_jsonl_gz(
        reviews_path,
        [
            {
                "parent_asin": "P1",
                "asin": "SKU1",
                "rating": 5,
                "text": "Battery life is excellent and lasts for days.",
                "title": "Great battery",
                "verified_purchase": True,
                "helpful_vote": 3,
            },
            {
                "parent_asin": "P2",
                "asin": "SKU2",
                "rating": 2,
                "text": "The screen is too dim and hard to read.",
                "title": "Bad screen",
                "verified_purchase": False,
                "helpful_vote": 1,
            },
        ],
    )

    parser = DataParser(str(raw_dir))
    products = list(parser.parse_metadata(str(metadata_path)))
    reviews = list(parser.parse_reviews(str(reviews_path)))
    selected_asins = {"P1", "P2"}

    db_manager = DatabaseManager(str(db_path))
    db_manager.create_tables()

    config_loader = ConfigLoader()
    brand_extractor = BrandExtractor("electronics", config_loader)
    loader = DataLoader(db_manager, "Electronics", brand_extractor)
    category_id = loader.load_category(
        {"category": {"amazon_category_id": "Electronics"}}
    )
    loader.load_brands(products)
    loader.load_products(products, selected_asins)
    loader.load_reviews(reviews, selected_asins, batch_size=10)
    loader.update_counts()

    with db_manager.get_session() as session:
        db_reviews = session.query(Review).all()
        assert len(db_reviews) == 2
        review_data = [{"id": r.id, "text": r.text} for r in db_reviews]

    processor = BatchProcessor(analyzer=FakeAnalyzer(), batch_size=2, checkpoint_interval=99)

    def callback(_batch_num, batch_results, _stats):
        with db_manager.get_session() as session:
            for result in batch_results:
                for aspect_data in result["aspects"]:
                    session.add(
                        AspectSentiment(
                            review_id=result["review_id"],
                            aspect_name=aspect_data["aspect"],
                            aspect_tier=aspect_data["tier"],
                            sentiment=aspect_data["sentiment"],
                            confidence_score=aspect_data["confidence"],
                            detection_method="integration-test",
                        )
                    )

    processor.process_reviews(review_data, callback=callback, checkpoint_name="mini_integration")

    with db_manager.get_session() as session:
        assert session.query(AspectSentiment).count() >= 2
        assert session.query(Review).count() == 2
        assert category_id is not None

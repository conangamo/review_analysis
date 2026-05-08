"""Quick sentiment tuning loop without writing to database."""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import func

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_engine import SentimentAnalyzer, ZeroShotClassifier
from src.core import AspectManager, ConfigLoader, get_env
from src.database.db_manager import DatabaseManager
from src.database.models import Category, Product, Review


def _safe_text(value: str) -> str:
    """Return terminal-safe text for current stdout encoding."""
    if value is None:
        return ""
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _resolve_analysis_settings(config_loader, category_config, env):
    """Resolve analyzer settings from config/env (same logic as run_analysis)."""
    model_config = config_loader.load_model_config()
    analyzer_config = model_config.get("sentiment_analyzer", {})
    optimization_config = model_config.get("optimization", {})

    return {
        "use_keyword_filter": env.get_bool(
            "USE_KEYWORD_FILTER",
            optimization_config.get("use_keyword_filter", True),
        ),
        "confidence_threshold": env.get_float(
            "CONFIDENCE_THRESHOLD",
            analyzer_config.get("confidence_threshold", 0.65),
        ),
        "min_confidence_tier1": env.get_float(
            "MIN_CONFIDENCE_TIER1",
            analyzer_config.get("min_confidence_tier1", 0.55),
        ),
        "min_confidence_tier2": env.get_float(
            "MIN_CONFIDENCE_TIER2",
            analyzer_config.get("min_confidence_tier2", 0.70),
        ),
        "min_confidence_neutral": env.get_float(
            "MIN_CONFIDENCE_NEUTRAL",
            analyzer_config.get("min_confidence_neutral", 0.40),
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Quick sentiment tuning on sampled reviews (no DB writes)."
    )
    parser.add_argument("--category", type=str, required=True, help="Category name")
    parser.add_argument(
        "--limit-reviews",
        type=int,
        default=200,
        help="Number of sampled reviews to analyze (default: 200)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for deterministic sampling",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=8,
        help="Number of sample outputs to print",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed blocks like inspect_recent_analysis.",
    )
    parser.add_argument(
        "--text-max",
        type=int,
        default=220,
        help="Max text chars in detailed preview (default: 220).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device override (default: auto)",
    )
    args = parser.parse_args()

    random.seed(args.random_seed)

    print("\n" + "=" * 70)
    print("Quick Sentiment Tuning (No DB writes)")
    print("=" * 70)
    print(f"Category: {args.category}")
    print(f"Sample size: {args.limit_reviews}")
    print()

    config_loader = ConfigLoader()
    env = get_env()
    category_config = config_loader.load_category_config(args.category)
    category_name = category_config["category"]["name"]
    settings = _resolve_analysis_settings(config_loader, category_config, env)

    db_path = env.get_path("DB_PATH", "./data/processed/reviews.db")
    db_manager = DatabaseManager(str(db_path))

    with db_manager.get_session() as session:
        category_obj = session.query(Category).filter_by(name=category_name).first()
        if not category_obj:
            print(f"ERROR: Category '{category_name}' not found in DB")
            return 1

        category_id = category_obj.id
        selected_asins = session.query(Product.parent_asin).filter_by(
            category_id=category_id,
            is_selected=True,
        ).all()
        selected_asins = [row[0] for row in selected_asins]
        if not selected_asins:
            print("ERROR: No selected products found")
            return 1

        # Random sample directly from DB for fast iteration
        sampled_reviews = (
            session.query(
                Review.id,
                Review.parent_asin,
                Review.rating,
                Review.title,
                Review.text,
            )
            .filter(Review.parent_asin.in_(selected_asins))
            .filter(func.length(Review.text) > 20)
            .order_by(func.random())
            .limit(args.limit_reviews)
            .all()
        )

    if not sampled_reviews:
        print("ERROR: No reviews sampled. Check parsed data first.")
        return 1

    use_gpu = env.get_bool("USE_GPU", True)
    use_fp16 = env.get_bool("USE_FP16", True)
    model_name = env.get_str("MODEL_NAME", "valhalla/distilbart-mnli-12-3")

    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda"
    else:
        device = "cuda" if use_gpu else "cpu"

    print("Initializing analyzer...")
    aspect_manager = AspectManager(args.category, config_loader)
    classifier = ZeroShotClassifier(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
        batch_size=env.get_int("BATCH_SIZE", 32),
    )
    analyzer = SentimentAnalyzer(
        category_name=args.category,
        aspect_manager=aspect_manager,
        classifier=classifier,
        use_keyword_filter=settings["use_keyword_filter"],
        confidence_threshold=settings["confidence_threshold"],
        min_confidence_tier1=settings["min_confidence_tier1"],
        min_confidence_tier2=settings["min_confidence_tier2"],
        min_confidence_neutral=settings["min_confidence_neutral"],
    )

    print(f"Analyzer ready on device: {device}")
    print()

    sentiment_counts = Counter()
    aspect_count_per_review = []
    analyzed_examples = []

    for review_id, parent_asin, rating, title, text in sampled_reviews:
        aspects = analyzer.analyze_review(text, review_id=review_id)
        aspect_count_per_review.append(len(aspects))
        for item in aspects:
            sentiment_counts[item["sentiment"]] += 1
        if aspects and len(analyzed_examples) < args.show_examples:
            analyzed_examples.append(
                {
                    "review_id": review_id,
                    "parent_asin": parent_asin,
                    "rating": rating,
                    "title": title,
                    "text": text,
                    "aspects": aspects,
                }
            )

    total_aspects = sum(sentiment_counts.values())
    neutral_pct = (
        sentiment_counts["neutral"] / total_aspects * 100 if total_aspects > 0 else 0.0
    )
    avg_aspects = (
        sum(aspect_count_per_review) / len(aspect_count_per_review)
        if aspect_count_per_review
        else 0.0
    )

    print("=" * 70)
    print("Quick Tuning Result")
    print("=" * 70)
    print(f"Sampled reviews: {len(sampled_reviews):,}")
    print(f"Total extracted aspects: {total_aspects:,}")
    print(f"Avg aspects/review: {avg_aspects:.2f}")
    print()
    print("Sentiment distribution:")
    for key in ("positive", "negative", "neutral"):
        count = sentiment_counts.get(key, 0)
        pct = (count / total_aspects * 100) if total_aspects > 0 else 0.0
        print(f"  - {key:8s}: {count:5d} ({pct:6.2f}%)")
    print(f"\nNeutral ratio (target sanity check): {neutral_pct:.2f}%")

    print("\nExamples:")
    for idx, sample in enumerate(analyzed_examples, start=1):
        review_id = sample["review_id"]
        parent_asin = sample["parent_asin"]
        rating = sample["rating"]
        title = (sample["title"] or "").strip()
        text = " ".join((sample["text"] or "").strip().split())
        aspects = sample["aspects"]

        if args.detailed:
            print("-" * 70)
            print(
                f"review_id={review_id}  parent_asin={parent_asin}  "
                f"rating={rating if rating is not None else 'N/A'}"
            )
            if title:
                print(f"title: {_safe_text(title[:120])}")
            preview = text[: args.text_max] + ("..." if len(text) > args.text_max else "")
            print(f"text: {_safe_text(preview)}")
            for a in aspects:
                print(
                    f"  - {a['aspect']:20s}  {a['sentiment']:8s}  "
                    f"conf={a['confidence']:.3f}  tier={a['tier']}"
                )
        else:
            preview = text[:140]
            labels = ", ".join(f"{a['aspect']}:{a['sentiment']}" for a in aspects[:5])
            print(f"{idx}. review_id={review_id} | {_safe_text(preview)}")
            print(f"   -> {_safe_text(labels)}")

    print("\nTip: tune config/code, rerun this script, then run full pipeline once.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


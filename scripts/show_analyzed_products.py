"""Show products that already have ABSA data."""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db_manager import DatabaseManager
from src.database.models import Review, AspectSentiment, Product, Category
from sqlalchemy import func, distinct


def main():
    """Show products with ABSA analysis rows."""
    parser = argparse.ArgumentParser(
        description="List products that already have ABSA data."
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional category name filter (e.g., electronics)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Max number of products to display",
    )
    args = parser.parse_args()

    db = DatabaseManager("data/processed/reviews.db")

    print("\n" + "=" * 80)
    print("Products with ABSA Data")
    print("=" * 80)
    print()

    with db.get_session() as session:
        query = (
            session.query(
                Product.parent_asin,
                Product.title,
                Product.average_rating,
                func.count(distinct(AspectSentiment.review_id)).label("analyzed_reviews"),
                func.count(AspectSentiment.id).label("total_aspects"),
            )
            .join(Review, Review.parent_asin == Product.parent_asin)
            .join(AspectSentiment, AspectSentiment.review_id == Review.id)
        )

        if args.category:
            category_name = args.category.strip().lower()
            query = query.join(Category, Category.id == Product.category_id).filter(
                func.lower(Category.name) == category_name
            )

        analyzed_products = (
            query.group_by(Product.parent_asin)
            .order_by(func.count(distinct(AspectSentiment.review_id)).desc())
            .limit(args.limit)
            .all()
        )

        if not analyzed_products:
            print("No products found with ABSA data.")
            if args.category:
                print(f"Category filter: {args.category}")
            print()
            return

        print(f"Showing {len(analyzed_products)} product(s) with ABSA data:\n")

        for i, (asin, title, rating, reviews, aspects) in enumerate(analyzed_products, 1):
            print(f"{i}. ASIN: {asin}")
            print(f"   Title: {title[:90]}{'...' if len(title) > 90 else ''}")
            print(f"   Rating: {rating:.1f}" if rating is not None else "   Rating: N/A")
            print(f"   Analyzed: {reviews} reviews, {aspects} aspects")
            print()

        print("-" * 80)
        print("\nAspect Sentiment Breakdown (global):\n")

        aspect_breakdown = session.query(
            AspectSentiment.aspect_name,
            AspectSentiment.sentiment,
            func.count(AspectSentiment.id).label("count"),
        ).group_by(
            AspectSentiment.aspect_name,
            AspectSentiment.sentiment,
        ).order_by(
            AspectSentiment.aspect_name,
            AspectSentiment.sentiment,
        ).all()

        current_aspect = None
        for aspect, sentiment, count in aspect_breakdown:
            if aspect != current_aspect:
                if current_aspect:
                    print()
                print(f"{aspect}:")
                current_aspect = aspect
            print(f"  {sentiment:8s}: {count:3d}")

        print("\n" + "=" * 80)
        print("Tip: Use one ASIN above in your Streamlit demo search.")
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()

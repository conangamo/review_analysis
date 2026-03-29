"""Show products that have been analyzed with AI."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db_manager import DatabaseManager
from src.database.models import Review, AspectSentiment, Product
from sqlalchemy import func, distinct

def main():
    """Show analyzed products."""
    
    db = DatabaseManager('data/processed/reviews.db')
    
    print("\n" + "="*70)
    print("🔍 Products with AI Analysis")
    print("="*70)
    print()
    
    with db.get_session() as session:
        # Find products that have aspect analysis
        analyzed_products = session.query(
            Product.parent_asin,
            Product.title,
            Product.average_rating,
            func.count(distinct(AspectSentiment.review_id)).label('analyzed_reviews'),
            func.count(AspectSentiment.id).label('total_aspects')
        ).join(
            Review, Review.parent_asin == Product.parent_asin
        ).join(
            AspectSentiment, AspectSentiment.review_id == Review.id
        ).group_by(
            Product.parent_asin
        ).order_by(
            func.count(distinct(AspectSentiment.review_id)).desc()
        ).all()
        
        print(f"Found {len(analyzed_products)} products with AI analysis:\n")
        
        for i, (asin, title, rating, reviews, aspects) in enumerate(analyzed_products, 1):
            print(f"{i}. ASIN: {asin}")
            print(f"   Title: {title[:60]}...")
            print(f"   Rating: {rating:.1f} ⭐")
            print(f"   Analyzed: {reviews} reviews, {aspects} aspects")
            print()
        
        # Show aspect breakdown
        print("-" * 70)
        print("\n📊 Aspect Sentiment Breakdown:\n")
        
        aspect_breakdown = session.query(
            AspectSentiment.aspect_name,
            AspectSentiment.sentiment,
            func.count(AspectSentiment.id).label('count')
        ).group_by(
            AspectSentiment.aspect_name,
            AspectSentiment.sentiment
        ).order_by(
            AspectSentiment.aspect_name,
            AspectSentiment.sentiment
        ).all()
        
        current_aspect = None
        for aspect, sentiment, count in aspect_breakdown:
            if aspect != current_aspect:
                if current_aspect:
                    print()
                print(f"{aspect}:")
                current_aspect = aspect
            print(f"  {sentiment:8s}: {count:3d}")
        
        print("\n" + "="*70)
        print("\n💡 Tip: To view these in UI, select the product ASIN in the sidebar")
        print("="*70)
        print()

if __name__ == "__main__":
    main()

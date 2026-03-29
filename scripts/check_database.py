"""Check database status and contents."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db_manager import DatabaseManager
from src.database.models import Category, Brand, Product, Review, AspectSentiment
from src.core import get_env


def main():
    """Check database contents."""
    
    print("\n" + "="*70)
    print("🔍 Database Status Check")
    print("="*70)
    
    env = get_env()
    db_path = env.get_path("DB_PATH", "./data/processed/reviews.db")
    
    db = DatabaseManager(str(db_path))
    
    # Print overall stats
    db.print_stats()
    
    # Detailed breakdown
    print("\n📊 Detailed Breakdown")
    print("="*70)
    
    with db.get_session() as session:
        # Categories
        categories = session.query(Category).all()
        print(f"\n📁 Categories ({len(categories)}):")
        for cat in categories:
            print(f"  • {cat.name}")
            print(f"    - Brands: {cat.total_brands:,}")
            print(f"    - Products: {cat.total_products:,}")
            print(f"    - Reviews: {cat.total_reviews:,}")
        
        # Top brands
        print(f"\n🏷️  Top 10 Brands (by product count):")
        top_brands = session.query(Brand).order_by(
            Brand.product_count.desc()
        ).limit(10).all()
        
        for i, brand in enumerate(top_brands, 1):
            print(f"  {i}. {brand.name:20s} - {brand.product_count:,} products")
        
        # Selected products
        selected_count = session.query(Product).filter_by(is_selected=True).count()
        print(f"\n📦 Selected Products: {selected_count:,}")
        
        # Reviews for selected products
        reviews_for_selected = session.query(Review).join(
            Product
        ).filter(
            Product.is_selected == True
        ).count()
        
        print(f"💬 Reviews for Selected Products: {reviews_for_selected:,}")
        
        # Check if any reviews exist
        total_reviews = session.query(Review).count()
        print(f"💬 Total Reviews in DB: {total_reviews:,}")
        
        if total_reviews > 0:
            # Sample review
            sample = session.query(Review).first()
            print(f"\n📝 Sample Review:")
            print(f"  Product: {sample.parent_asin}")
            print(f"  Rating: {sample.rating} ⭐")
            print(f"  Text: {sample.text[:100]}...")
        
        # Aspect sentiments
        aspect_count = session.query(AspectSentiment).count()
        print(f"\n🎯 Aspect Sentiments Analyzed: {aspect_count:,}")
        
        if aspect_count > 0:
            # Sample aspect
            sample_aspect = session.query(AspectSentiment).first()
            print(f"\n🔍 Sample Aspect Analysis:")
            print(f"  Review ID: {sample_aspect.review_id}")
            print(f"  Aspect: {sample_aspect.aspect_name}")
            print(f"  Sentiment: {sample_aspect.sentiment}")
            print(f"  Confidence: {sample_aspect.confidence_score:.2%}")
    
    print("\n" + "="*70)
    print("✅ Check Complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

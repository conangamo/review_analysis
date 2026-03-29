"""Generate product and brand summaries from analysis results."""

import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import ConfigLoader, get_env
from src.database.db_manager import DatabaseManager
from src.database.models import (
    Category, Product, Review, AspectSentiment,
    ProductSummary, BrandSummary, ProcessingStatus
)
from sqlalchemy import func

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main summary generation function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Generate product and brand summaries"
    )
    parser.add_argument(
        '--category',
        type=str,
        required=True,
        help='Category name (e.g., electronics)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate all summaries (even if exist)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("📊 Summary Generation Pipeline")
    print("="*70)
    print(f"Category: {args.category}")
    print()
    
    try:
        # Step 1: Initialize
        print("📋 Step 1: Initializing...")
        print("-" * 70)
        
        config_loader = ConfigLoader()
        env = get_env()
        
        category_config = config_loader.load_category_config(args.category)
        category_name = category_config['category']['name']
        
        db_path = env.get_path("DB_PATH", "./data/processed/reviews.db")
        db_manager = DatabaseManager(str(db_path))
        
        print(f"✅ Category: {category_name}")
        print(f"✅ Database: {db_path}")
        print()
        
        # Get category ID
        with db_manager.get_session() as session:
            category_obj = session.query(Category).filter_by(name=category_name).first()
            
            if not category_obj:
                print(f"❌ Error: Category '{category_name}' not found")
                return 1
            
            category_id = category_obj.id
        
        # Step 2: Generate product summaries
        print("📦 Step 2: Generating product summaries...")
        print("-" * 70)
        
        with db_manager.get_session() as session:
            # Get all selected products
            products = session.query(Product).filter_by(
                category_id=category_id,
                is_selected=True
            ).all()
            
            if not products:
                print(f"❌ Error: No selected products found")
                return 1
            
            print(f"Found {len(products):,} selected products")
            print()
            
            # Process each product
            for i, product in enumerate(products, 1):
                if i % 100 == 0:
                    print(f"Progress: {i}/{len(products)} products...")
                
                # Skip if already has summary and not forcing
                if not args.force:
                    existing = session.query(ProductSummary).filter_by(
                        parent_asin=product.parent_asin
                    ).first()
                    if existing:
                        continue
                
                # Generate summary
                summary = generate_product_summary(session, product)
                
                if summary:
                    session.merge(summary)
                    
                    # Commit every 50 products
                    if i % 50 == 0:
                        session.commit()
            
            # Final commit
            session.commit()
        
        print(f"✅ Product summaries generated")
        print()
        
        # Step 3: Generate brand summaries
        print("🏷️  Step 3: Generating brand summaries...")
        print("-" * 70)
        
        with db_manager.get_session() as session:
            # Get all brands with products
            from src.database.models import Brand
            
            brands = session.query(Brand).filter_by(
                category_id=category_id
            ).filter(
                Brand.product_count > 0
            ).all()
            
            print(f"Found {len(brands):,} brands")
            print()
            
            # Process each brand
            for i, brand in enumerate(brands, 1):
                if i % 20 == 0:
                    print(f"Progress: {i}/{len(brands)} brands...")
                
                # Skip if already has summary and not forcing
                if not args.force:
                    existing = session.query(BrandSummary).filter_by(
                        brand_id=brand.id
                    ).first()
                    if existing:
                        continue
                
                # Generate summary
                summary = generate_brand_summary(session, brand, category_id)
                
                if summary:
                    session.merge(summary)
            
            # Commit
            session.commit()
        
        print(f"✅ Brand summaries generated")
        print()
        
        # Step 4: Statistics
        print("="*70)
        print("🎉 Summary Generation Complete!")
        print("="*70)
        
        db_manager.print_stats()
        
        print("\n" + "="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print("\nNext steps:")
        print(f"  Launch UI: streamlit run src/ui/app.py")
        print()
        
        return 0
    
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        logger.exception("Error during summary generation")
        return 1


def generate_product_summary(session, product: Product) -> ProductSummary:
    """
    Generate summary for a product.
    
    Args:
        session: Database session
        product: Product object
    
    Returns:
        ProductSummary object
    """
    # Get all reviews for this product
    reviews = session.query(Review).filter_by(
        parent_asin=product.parent_asin
    ).all()
    
    if not reviews:
        return None
    
    # Calculate rating distribution
    rating_dist = defaultdict(int)
    for review in reviews:
        rating_dist[int(review.rating)] += 1
    
    # Get aspect sentiments
    aspect_sentiments = session.query(AspectSentiment).join(
        Review
    ).filter(
        Review.parent_asin == product.parent_asin
    ).all()
    
    # Aggregate by aspect
    aspect_summary = defaultdict(lambda: {
        'total_mentions': 0,
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'confidences': []
    })
    
    overall_sentiment = defaultdict(int)
    
    for sentiment in aspect_sentiments:
        aspect = sentiment.aspect_name
        aspect_summary[aspect]['total_mentions'] += 1
        aspect_summary[aspect][sentiment.sentiment] += 1
        aspect_summary[aspect]['confidences'].append(sentiment.confidence_score)
        
        # Track overall sentiment
        overall_sentiment[sentiment.sentiment] += 1
    
    # Calculate percentages and averages
    for aspect, data in aspect_summary.items():
        total = data['total_mentions']
        data['positive_pct'] = (data['positive'] / total * 100) if total > 0 else 0
        data['negative_pct'] = (data['negative'] / total * 100) if total > 0 else 0
        data['neutral_pct'] = (data['neutral'] / total * 100) if total > 0 else 0
        data['avg_confidence'] = sum(data['confidences']) / len(data['confidences']) if data['confidences'] else 0
        del data['confidences']  # Remove raw list
    
    # Select representative reviews
    top_positive = select_representative_reviews(session, product.parent_asin, 'positive', limit=5)
    top_negative = select_representative_reviews(session, product.parent_asin, 'negative', limit=5)
    top_mixed = select_representative_reviews(session, product.parent_asin, 'neutral', limit=3)
    
    # Create summary
    summary = ProductSummary(
        parent_asin=product.parent_asin,
        category_id=product.category_id,
        total_reviews=len(reviews),
        avg_rating=product.average_rating,
        rating_distribution=json.dumps(dict(rating_dist)),
        overall_positive=overall_sentiment.get('positive', 0),
        overall_negative=overall_sentiment.get('negative', 0),
        overall_neutral=overall_sentiment.get('neutral', 0),
        aspects_summary=json.dumps(dict(aspect_summary)),
        top_positive_review_ids=json.dumps(top_positive),
        top_negative_review_ids=json.dumps(top_negative),
        top_mixed_review_ids=json.dumps(top_mixed),
        aspects_analyzed=json.dumps(list(aspect_summary.keys())),
        last_updated=datetime.now()
    )
    
    return summary


def generate_brand_summary(session, brand, category_id: int) -> BrandSummary:
    """
    Generate summary for a brand.
    
    Args:
        session: Database session
        brand: Brand object
        category_id: Category ID
    
    Returns:
        BrandSummary object
    """
    # Get all products for this brand
    products = session.query(Product).filter_by(
        brand_id=brand.id,
        category_id=category_id
    ).all()
    
    if not products:
        return None
    
    product_asins = [p.parent_asin for p in products]
    
    # Get total reviews
    total_reviews = session.query(func.count(Review.id)).filter(
        Review.parent_asin.in_(product_asins)
    ).scalar() or 0
    
    # Calculate average rating
    avg_rating = session.query(func.avg(Product.average_rating)).filter(
        Product.parent_asin.in_(product_asins)
    ).scalar()
    
    # Aggregate aspect sentiments across all products
    aspect_sentiments = session.query(AspectSentiment).join(
        Review
    ).filter(
        Review.parent_asin.in_(product_asins)
    ).all()
    
    # Aggregate by aspect
    aspect_summary = defaultdict(lambda: {
        'total_mentions': 0,
        'positive': 0,
        'negative': 0,
        'neutral': 0
    })
    
    for sentiment in aspect_sentiments:
        aspect = sentiment.aspect_name
        aspect_summary[aspect]['total_mentions'] += 1
        aspect_summary[aspect][sentiment.sentiment] += 1
    
    # Calculate percentages
    for aspect, data in aspect_summary.items():
        total = data['total_mentions']
        data['positive_pct'] = (data['positive'] / total * 100) if total > 0 else 0
        data['negative_pct'] = (data['negative'] / total * 100) if total > 0 else 0
        data['neutral_pct'] = (data['neutral'] / total * 100) if total > 0 else 0
    
    # Create summary
    summary = BrandSummary(
        brand_id=brand.id,
        category_id=category_id,
        total_products=len(products),
        total_reviews=total_reviews,
        avg_rating=float(avg_rating) if avg_rating else None,
        aspects_summary=json.dumps(dict(aspect_summary)),
        last_updated=datetime.now()
    )
    
    return summary


def select_representative_reviews(
    session,
    parent_asin: str,
    sentiment_type: str,
    limit: int = 5
) -> list:
    """
    Select representative reviews for a product.
    
    Args:
        session: Database session
        parent_asin: Product ASIN
        sentiment_type: 'positive', 'negative', or 'neutral'
        limit: Number of reviews to select
    
    Returns:
        List of review IDs
    """
    # Get reviews with this sentiment type
    reviews = session.query(Review).join(
        AspectSentiment
    ).filter(
        Review.parent_asin == parent_asin,
        AspectSentiment.sentiment == sentiment_type
    ).order_by(
        Review.helpful_vote.desc(),  # Most helpful first
        AspectSentiment.confidence_score.desc()  # Most confident
    ).limit(limit).all()
    
    return [r.id for r in reviews]


if __name__ == "__main__":
    sys.exit(main())

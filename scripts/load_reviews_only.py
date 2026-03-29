"""Load reviews for selected products (skip products parsing)."""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import ConfigLoader, get_env
from src.database.db_manager import DatabaseManager
from src.database.models import Product, Review
from src.data_processing import DataParser, DataLoader
from src.core.brand_extractor import BrandExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Load reviews only for already selected products."""
    
    parser = argparse.ArgumentParser(
        description="Load reviews for selected products"
    )
    parser.add_argument(
        '--category',
        type=str,
        required=True,
        help='Category name (e.g., electronics)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for loading reviews'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("💬 Load Reviews for Selected Products")
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
        
        data_dir = Path(f"./data/raw/{args.category.lower()}")
        
        print(f"✅ Category: {category_name}")
        print(f"✅ Database: {db_path}")
        print(f"✅ Data dir: {data_dir}")
        print()
        
        # Step 2: Get selected products
        print("📦 Step 2: Loading selected products...")
        print("-" * 70)
        
        with db_manager.get_session() as session:
            from src.database.models import Category
            category_obj = session.query(Category).filter_by(name=category_name).first()
            
            if not category_obj:
                print(f"❌ Error: Category not found")
                return 1
            
            category_id = category_obj.id
            
            # Get selected products
            selected_products = session.query(Product).filter_by(
                category_id=category_id,
                is_selected=True
            ).all()
            
            if not selected_products:
                print(f"❌ Error: No selected products found")
                print(f"   Run: python scripts/parse_data.py --category {args.category}")
                return 1
            
            selected_asins = {p.parent_asin for p in selected_products}
        
        print(f"✅ Found {len(selected_asins):,} selected products")
        print()
        
        # Step 3: Parse reviews
        print("💬 Step 3: Parsing reviews from file...")
        print("-" * 70)
        
        reviews_file = data_dir / f"{category_config['category']['amazon_category_id']}.jsonl.gz"
        
        if not reviews_file.exists():
            print(f"❌ Error: Reviews file not found: {reviews_file}")
            return 1
        
        print(f"File: {reviews_file.name}")
        print(f"Parsing all reviews and filtering for {len(selected_asins):,} products...")
        print("This may take a while (5-15 minutes)...")
        print()
        
        parser = DataParser(str(data_dir))
        
        # Step 4: Initialize loader
        print("💾 Step 4: Setting up loader...")
        print("-" * 70)
        
        brand_extractor = BrandExtractor(args.category, config_loader)
        loader = DataLoader(db_manager, category_name, brand_extractor)
        loader._category_id = category_id
        
        print(f"✅ Loader ready with batch size: {args.batch_size}")
        print()
        
        # Step 5: Stream and load reviews (MEMORY EFFICIENT!)
        print("💬 Step 5: Streaming and loading reviews...")
        print("-" * 70)
        print("Processing in streaming mode (memory efficient)...")
        print()
        
        batch = []
        total_parsed = 0
        total_loaded = 0
        
        with db_manager.get_session() as session:
            for review in parser.parse_reviews(str(reviews_file)):
                total_parsed += 1
                
                # Only process reviews for selected products
                if review['parent_asin'] in selected_asins:
                    batch.append(review)
                    
                    # Insert batch when full
                    if len(batch) >= args.batch_size:
                        loader._insert_review_batch(session, batch)
                        total_loaded += len(batch)
                        batch = []
                
                # Progress update
                if total_parsed % 500000 == 0:
                    print(f"  Parsed: {total_parsed:,} | Loaded: {total_loaded:,}")
            
            # Insert remaining batch
            if batch:
                loader._insert_review_batch(session, batch)
                total_loaded += len(batch)
        
        print(f"\n✅ Parsed {total_parsed:,} total reviews")
        print(f"✅ Loaded {total_loaded:,} reviews for selected products")
        print()
        
        if total_loaded == 0:
            print("⚠️  No reviews found for selected products!")
            print("This might mean:")
            print("  1. Selected products don't have reviews in the dataset")
            print("  2. Parent ASIN mismatch between products and reviews")
            return 1
        
        # Update loader stats
        loader.stats['reviews_created'] = total_loaded
        
        print()
        
        # Step 6: Update counts
        print("🔢 Step 5: Updating counts...")
        print("-" * 70)
        
        loader.update_counts()
        
        print(f"✅ Counts updated")
        print()
        
        # Final stats
        print("="*70)
        print("🎉 Reviews Loaded Successfully!")
        print("="*70)
        
        loader.print_stats()
        db_manager.print_stats()
        
        print("\n" + "="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print("\nNext steps:")
        print(f"  1. Run analysis: python scripts/run_analysis.py --category {args.category}")
        print(f"  2. Generate summaries: python scripts/generate_summaries.py --category {args.category}")
        print(f"  3. Launch UI: streamlit run src/ui/app.py")
        print()
        
        return 0
    
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        logger.exception("Error loading reviews")
        return 1


if __name__ == "__main__":
    sys.exit(main())

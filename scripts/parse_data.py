"""Parse and load Amazon Reviews data into database."""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import ConfigLoader, BrandExtractor, get_env
from src.database.db_manager import DatabaseManager
from src.data_processing import DataParser, ProductSampler, DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main data parsing and loading function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Parse and load Amazon Reviews data"
    )
    parser.add_argument(
        '--category',
        type=str,
        required=True,
        help='Category name (e.g., electronics)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Data directory (default: ./data/raw/{category})'
    )
    parser.add_argument(
        '--limit-products',
        type=int,
        help='Limit number of products to parse (for testing)'
    )
    parser.add_argument(
        '--limit-reviews',
        type=int,
        help='Limit number of reviews to parse (for testing)'
    )
    parser.add_argument(
        '--skip-reviews',
        action='store_true',
        help='Skip loading reviews (only load products)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("📊 Amazon Reviews Data Parser")
    print("="*70)
    print(f"Category: {args.category}")
    print()
    
    try:
        # Step 1: Load configuration
        print("📋 Step 1: Loading configuration...")
        print("-" * 70)
        
        config_loader = ConfigLoader()
        env = get_env()
        
        # Load category config
        category_config = config_loader.load_category_config(args.category)
        category_name = category_config['category']['name']
        
        print(f"✅ Category: {category_name}")
        print(f"✅ Amazon ID: {category_config['category']['amazon_category_id']}")
        
        # Get data directory
        if args.data_dir:
            data_dir = Path(args.data_dir)
        else:
            data_dir = Path(f"./data/raw/{args.category.lower()}")
        
        if not data_dir.exists():
            print(f"\n❌ Error: Data directory not found: {data_dir}")
            print(f"   Run: python scripts/download_data.py --category {args.category}")
            return 1
        
        print(f"✅ Data directory: {data_dir}")
        print()
        
        # Step 2: Initialize components
        print("🔧 Step 2: Initializing components...")
        print("-" * 70)
        
        # Database
        db_path = env.get_path("DB_PATH", "./data/processed/reviews.db")
        db_manager = DatabaseManager(str(db_path))
        print(f"✅ Database: {db_path}")
        
        # Parser
        data_parser = DataParser(str(data_dir))
        print(f"✅ Parser initialized")
        
        # Brand extractor
        brand_extractor = BrandExtractor(args.category, config_loader)
        print(f"✅ Brand extractor initialized")
        
        # Sampler
        min_reviews = category_config['data'].get('min_reviews', 20)
        sampler = ProductSampler(min_reviews=min_reviews)
        print(f"✅ Sampler initialized (min_reviews: {min_reviews})")
        
        # Loader
        loader = DataLoader(db_manager, category_name, brand_extractor)
        print(f"✅ Loader initialized")
        print()
        
        # Step 3: Load category
        print("📁 Step 3: Setting up category...")
        print("-" * 70)
        
        category_id = loader.load_category(category_config)
        print(f"✅ Category loaded (ID: {category_id})")
        print()
        
        # Step 4: Parse metadata
        print("📦 Step 4: Parsing product metadata...")
        print("-" * 70)
        
        metadata_file = data_dir / f"meta_{category_config['category']['amazon_category_id']}.jsonl.gz"
        
        if not metadata_file.exists():
            print(f"❌ Error: Metadata file not found: {metadata_file}")
            return 1
        
        print(f"File: {metadata_file.name}")
        
        # Parse all products into memory (needed for sampling)
        products = list(data_parser.parse_metadata(
            str(metadata_file),
            limit=args.limit_products
        ))
        
        print(f"✅ Parsed {len(products):,} products")
        print()
        
        # Step 5: Sample products
        print("🎯 Step 5: Sampling products...")
        print("-" * 70)
        
        rating_bins = category_config['data'].get('rating_bins', [])
        
        if rating_bins:
            # Convert to tuples
            rating_bins = [tuple(bin_spec) for bin_spec in rating_bins]
            
            print(f"Strategy: Stratified sampling")
            print(f"Bins: {len(rating_bins)}")
            for min_r, max_r, count in rating_bins:
                print(f"  [{min_r}-{max_r}]: {count} products")
            
            selected_asins = sampler.stratified_sample(products, rating_bins)
        else:
            # Fallback to top-N
            top_n = category_config['data'].get('top_products', 3000)
            print(f"Strategy: Top-{top_n} sampling")
            
            selected_asins = sampler.top_n_sample(products, n=top_n)
        
        selected_asins = set(selected_asins)
        
        print(f"✅ Selected {len(selected_asins):,} products")
        sampler.print_stats()
        
        # Step 6: Load brands
        print("🏷️  Step 6: Loading brands...")
        print("-" * 70)
        
        loader.load_brands(products)
        print(f"✅ Loaded {len(loader._brand_cache)} unique brands")
        print()
        
        # Step 7: Load products
        print("📦 Step 7: Loading products into database...")
        print("-" * 70)
        
        loader.load_products(products, selected_asins)
        print(f"✅ Products loaded successfully")
        print()
        
        # Step 8: Parse and load reviews
        if not args.skip_reviews:
            print("💬 Step 8: Parsing and loading reviews...")
            print("-" * 70)
            
            reviews_file = data_dir / f"{category_config['category']['amazon_category_id']}.jsonl.gz"
            
            if not reviews_file.exists():
                print(f"❌ Error: Reviews file not found: {reviews_file}")
                return 1
            
            print(f"File: {reviews_file.name}")
            print(f"Loading reviews for {len(selected_asins):,} selected products...")
            print()
            
            # Parse reviews
            print("Parsing reviews (this may take a while)...")
            reviews = data_parser.parse_reviews(
                str(reviews_file),
                limit=args.limit_reviews
            )
            
            # Load in batches
            batch_size = env.get_int("PARSE_BATCH_SIZE", 1000)
            
            # Convert iterator to list for selected products only
            reviews_to_load = []
            for review in reviews:
                if review['parent_asin'] in selected_asins:
                    reviews_to_load.append(review)
                    
                    # Show progress every 100K reviews
                    if len(reviews_to_load) % 100000 == 0:
                        print(f"  Collected {len(reviews_to_load):,} reviews...")
            
            print(f"✅ Collected {len(reviews_to_load):,} reviews for selected products")
            print()
            
            # Load into database
            print("Loading reviews into database...")
            loader.load_reviews(reviews_to_load, selected_asins, batch_size)
            print(f"✅ Reviews loaded successfully")
            print()
        else:
            print("⏭️  Step 8: Skipping reviews (--skip-reviews flag)")
            print()
        
        # Step 9: Update counts
        print("🔢 Step 9: Updating counts...")
        print("-" * 70)
        
        loader.update_counts()
        print(f"✅ Counts updated")
        print()
        
        # Step 10: Final statistics
        print("="*70)
        print("🎉 Data Loading Complete!")
        print("="*70)
        
        loader.print_stats()
        
        # Database stats
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
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        logger.exception("Error during data loading")
        return 1


if __name__ == "__main__":
    sys.exit(main())

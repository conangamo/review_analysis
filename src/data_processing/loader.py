"""Load parsed data into database."""

import logging
from typing import List, Dict, Any, Set
from pathlib import Path
from sqlalchemy import text

try:
    from ..database.db_manager import DatabaseManager
    from ..database.models import Category, Brand, Product, Review
    from ..core.brand_extractor import BrandExtractor
    from ..core import get_env
except ImportError:
    # For direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.database.db_manager import DatabaseManager
    from src.database.models import Category, Brand, Product, Review
    from src.core.brand_extractor import BrandExtractor
    from src.core import get_env

logger = logging.getLogger(__name__)


class DataLoader:
    """Load data into database."""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        category_name: str,
        brand_extractor: BrandExtractor
    ):
        """
        Initialize data loader.
        
        Args:
            db_manager: Database manager instance
            category_name: Category name
            brand_extractor: Brand extractor instance
        """
        self.db = db_manager
        self.category_name = category_name
        self.brand_extractor = brand_extractor
        
        self.stats = {
            'brands_created': 0,
            'brands_updated': 0,
            'products_created': 0,
            'products_updated': 0,
            'reviews_created': 0,
            'reviews_skipped': 0,
            'errors': 0
        }
        
        # Cache for brand IDs
        self._brand_cache = {}
        self._category_id = None
    
    def load_category(self, category_config: Dict[str, Any]):
        """
        Load or update category in database.
        
        Args:
            category_config: Category configuration dictionary
        
        Returns:
            Category ID
        """
        with self.db.get_session() as session:
            # Check if category exists
            category = session.query(Category).filter_by(
                name=self.category_name
            ).first()
            
            if category:
                logger.info(f"Category '{self.category_name}' already exists")
                self._category_id = category.id
            else:
                # Create new category
                category = Category(
                    name=self.category_name,
                    amazon_id=category_config['category']['amazon_category_id'],
                    config_path=f"config/categories/{self.category_name.lower()}.yaml"
                )
                session.add(category)
                session.flush()
                
                self._category_id = category.id
                logger.info(f"Created category: {self.category_name} (ID: {self._category_id})")
        
        return self._category_id
    
    def load_brands(self, products: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Extract and load brands from products.
        
        Args:
            products: List of product dictionaries
        
        Returns:
            Dictionary mapping brand names to IDs
        """
        logger.info("Extracting brands from products...")
        
        # Extract unique brands
        brand_names = set()
        for product in products:
            brand_name = self.brand_extractor.extract_brand(product)
            if brand_name and brand_name != "Unknown":
                brand_names.add(brand_name)
        
        logger.info(f"Found {len(brand_names)} unique brands")
        
        # Load into database
        with self.db.get_session() as session:
            for brand_name in brand_names:
                normalized_name = self.brand_extractor.normalize_brand(brand_name)
                
                # Check if exists
                brand = session.query(Brand).filter_by(
                    normalized_name=normalized_name,
                    category_id=self._category_id
                ).first()
                
                if brand:
                    self.stats['brands_updated'] += 1
                else:
                    # Create new brand
                    brand = Brand(
                        name=brand_name,
                        normalized_name=normalized_name,
                        category_id=self._category_id
                    )
                    session.add(brand)
                    session.flush()
                    self.stats['brands_created'] += 1
                
                # Cache brand ID
                self._brand_cache[normalized_name] = brand.id
        
        logger.info(f"Loaded {len(self._brand_cache)} brands "
                   f"(created: {self.stats['brands_created']}, "
                   f"updated: {self.stats['brands_updated']})")
        
        return self._brand_cache
    
    def load_products(
        self,
        products: List[Dict[str, Any]],
        selected_asins: Set[str]
    ):
        """
        Load products into database.
        
        Args:
            products: List of product dictionaries
            selected_asins: Set of ASINs to mark as selected
        """
        logger.info(f"Loading {len(products)} products...")
        
        with self.db.get_session() as session:
            for product in products:
                try:
                    parent_asin = product['parent_asin']
                    
                    # Extract brand
                    brand_name = self.brand_extractor.extract_brand(product)
                    normalized_brand = self.brand_extractor.normalize_brand(brand_name)
                    brand_id = self._brand_cache.get(normalized_brand)
                    
                    # Check if product exists
                    existing = session.query(Product).filter_by(
                        parent_asin=parent_asin
                    ).first()
                    
                    if existing:
                        # Update existing
                        existing.title = product['title']
                        existing.brand_id = brand_id
                        existing.average_rating = product.get('average_rating')
                        existing.rating_number = product.get('rating_number', 0)
                        existing.price = product.get('price')
                        existing.image_url = product.get('image_url')
                        existing.features = product.get('features')
                        existing.description = product.get('description')
                        existing.product_metadata = product.get('product_metadata')
                        existing.is_selected = parent_asin in selected_asins
                        
                        self.stats['products_updated'] += 1
                    else:
                        # Create new
                        new_product = Product(
                            parent_asin=parent_asin,
                            title=product['title'],
                            brand_id=brand_id,
                            category_id=self._category_id,
                            average_rating=product.get('average_rating'),
                            rating_number=product.get('rating_number', 0),
                            price=product.get('price'),
                            image_url=product.get('image_url'),
                            features=product.get('features'),
                            description=product.get('description'),
                            product_metadata=product.get('product_metadata'),
                            is_selected=parent_asin in selected_asins,
                            selection_strategy='stratified_top' if parent_asin in selected_asins else None
                        )
                        session.add(new_product)
                        self.stats['products_created'] += 1
                    
                    # Commit every 1000 products
                    if (self.stats['products_created'] + self.stats['products_updated']) % 1000 == 0:
                        session.commit()
                        logger.info(f"Progress: {self.stats['products_created'] + self.stats['products_updated']} products")
                
                except Exception as e:
                    logger.error(f"Error loading product {product.get('parent_asin')}: {e}")
                    self.stats['errors'] += 1
                    continue
        
        logger.info(f"Loaded products: created {self.stats['products_created']}, "
                   f"updated {self.stats['products_updated']}")
    
    def load_reviews(
        self,
        reviews: List[Dict[str, Any]],
        selected_asins: Set[str],
        batch_size: int = 1000
    ):
        """
        Load reviews into database in batches.
        
        Args:
            reviews: List of review dictionaries
            selected_asins: Set of ASINs to load reviews for
            batch_size: Number of reviews per batch
        """
        logger.info(f"Loading reviews (batch size: {batch_size})...")
        
        batch = []
        total_processed = 0
        
        with self.db.get_session() as session:
            for review in reviews:
                # Only load reviews for selected products
                if review['parent_asin'] not in selected_asins:
                    self.stats['reviews_skipped'] += 1
                    continue
                
                try:
                    # Add to batch
                    batch.append(review)
                    
                    # Insert batch when full
                    if len(batch) >= batch_size:
                        self._insert_review_batch(session, batch)
                        total_processed += len(batch)
                        batch = []
                        
                        if total_processed % 10000 == 0:
                            logger.info(f"Progress: {total_processed:,} reviews loaded")
                
                except Exception as e:
                    logger.error(f"Error loading review: {e}")
                    self.stats['errors'] += 1
                    continue
            
            # Insert remaining batch
            if batch:
                self._insert_review_batch(session, batch)
                total_processed += len(batch)
        
        logger.info(f"Loaded {self.stats['reviews_created']:,} reviews, "
                   f"skipped {self.stats['reviews_skipped']:,}")
    
    def _insert_review_batch(self, session, batch: List[Dict[str, Any]]):
        """Insert a batch of reviews."""
        try:
            # Use bulk insert for better performance
            session.bulk_insert_mappings(Review, batch)
            session.commit()
            self.stats['reviews_created'] += len(batch)
        except Exception as e:
            logger.error(f"Error inserting review batch: {e}")
            session.rollback()
            
            # Try inserting one by one
            for review_data in batch:
                try:
                    review = Review(**review_data)
                    session.add(review)
                    session.commit()
                    self.stats['reviews_created'] += 1
                except Exception as e2:
                    logger.error(f"Error inserting individual review: {e2}")
                    session.rollback()
                    self.stats['errors'] += 1
    
    def update_counts(self):
        """Update product and brand counts."""
        logger.info("Updating product and brand counts...")
        
        with self.db.get_session() as session:
            # Update brand product counts
            session.execute(text("""
                UPDATE brands
                SET product_count = (
                    SELECT COUNT(*)
                    FROM products
                    WHERE products.brand_id = brands.id
                )
            """))
            
            # Update brand total reviews
            session.execute(text("""
                UPDATE brands
                SET total_reviews = (
                    SELECT COUNT(*)
                    FROM reviews
                    JOIN products ON reviews.parent_asin = products.parent_asin
                    WHERE products.brand_id = brands.id
                )
            """))
            
            # Update category counts
            session.execute(text("""
                UPDATE categories
                SET total_brands = (
                    SELECT COUNT(*)
                    FROM brands
                    WHERE brands.category_id = categories.id
                ),
                total_products = (
                    SELECT COUNT(*)
                    FROM products
                    WHERE products.category_id = categories.id
                ),
                total_reviews = (
                    SELECT COUNT(*)
                    FROM reviews
                    JOIN products ON reviews.parent_asin = products.parent_asin
                    WHERE products.category_id = categories.id
                )
                WHERE categories.id = :category_id
            """), {'category_id': self._category_id})
            
            session.commit()
        
        logger.info("Counts updated successfully")
    
    def get_stats(self) -> Dict[str, int]:
        """Get loading statistics."""
        return self.stats.copy()
    
    def print_stats(self):
        """Print loading statistics."""
        print("\n" + "="*60)
        print("📊 Data Loading Statistics")
        print("="*60)
        print(f"Brands:")
        print(f"  Created: {self.stats['brands_created']}")
        print(f"  Updated: {self.stats['brands_updated']}")
        print(f"\nProducts:")
        print(f"  Created: {self.stats['products_created']}")
        print(f"  Updated: {self.stats['products_updated']}")
        print(f"\nReviews:")
        print(f"  Created: {self.stats['reviews_created']:,}")
        print(f"  Skipped: {self.stats['reviews_skipped']:,}")
        print(f"\nErrors: {self.stats['errors']}")
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("DataLoader utility - use via scripts/parse_data.py")
    print("\nThis module provides:")
    print("  - load_category(): Load category into database")
    print("  - load_brands(): Extract and load brands")
    print("  - load_products(): Load product metadata")
    print("  - load_reviews(): Load reviews in batches")
    print("  - update_counts(): Update counts in database")

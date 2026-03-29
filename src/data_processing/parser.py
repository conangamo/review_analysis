"""Parse Amazon Reviews JSON data."""

import gzip
import json
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DataParser:
    """Parse Amazon Reviews 2023 dataset."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize parser.
        
        Args:
            data_dir: Directory containing raw data files
        """
        if data_dir is None:
            data_dir = "./data/raw"
        
        self.data_dir = Path(data_dir)
        self.stats = {
            'reviews_parsed': 0,
            'reviews_skipped': 0,
            'products_parsed': 0,
            'products_skipped': 0,
            'errors': 0
        }
    
    def parse_reviews(
        self, 
        file_path: str,
        limit: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Parse reviews from JSONL.GZ file.
        
        Args:
            file_path: Path to reviews file (.jsonl.gz)
            limit: Maximum number of reviews to parse (for testing)
        
        Yields:
            Dictionary containing review data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Review file not found: {file_path}")
        
        logger.info(f"Parsing reviews from: {file_path}")
        
        count = 0
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Check limit
                    if limit and count >= limit:
                        logger.info(f"Reached limit of {limit} reviews")
                        break
                    
                    try:
                        # Parse JSON
                        review = json.loads(line.strip())
                        
                        # Validate required fields
                        if not self._validate_review(review):
                            self.stats['reviews_skipped'] += 1
                            continue
                        
                        # Extract and clean data
                        cleaned_review = self._clean_review(review)
                        
                        self.stats['reviews_parsed'] += 1
                        count += 1
                        
                        yield cleaned_review
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                        self.stats['errors'] += 1
                        continue
                    
                    except Exception as e:
                        logger.error(f"Line {line_num}: Error parsing review - {e}")
                        self.stats['errors'] += 1
                        continue
        
        except Exception as e:
            logger.error(f"Error opening file {file_path}: {e}")
            raise
        
        logger.info(f"Parsed {self.stats['reviews_parsed']} reviews, "
                   f"skipped {self.stats['reviews_skipped']}, "
                   f"errors {self.stats['errors']}")
    
    def parse_metadata(
        self,
        file_path: str,
        limit: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Parse product metadata from JSONL.GZ file.
        
        Args:
            file_path: Path to metadata file (.jsonl.gz)
            limit: Maximum number of products to parse (for testing)
        
        Yields:
            Dictionary containing product metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {file_path}")
        
        logger.info(f"Parsing metadata from: {file_path}")
        
        count = 0
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Check limit
                    if limit and count >= limit:
                        logger.info(f"Reached limit of {limit} products")
                        break
                    
                    try:
                        # Parse JSON
                        product = json.loads(line.strip())
                        
                        # Validate required fields
                        if not self._validate_product(product):
                            self.stats['products_skipped'] += 1
                            continue
                        
                        # Extract and clean data
                        cleaned_product = self._clean_product(product)
                        
                        self.stats['products_parsed'] += 1
                        count += 1
                        
                        yield cleaned_product
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                        self.stats['errors'] += 1
                        continue
                    
                    except Exception as e:
                        logger.error(f"Line {line_num}: Error parsing product - {e}")
                        self.stats['errors'] += 1
                        continue
        
        except Exception as e:
            logger.error(f"Error opening file {file_path}: {e}")
            raise
        
        logger.info(f"Parsed {self.stats['products_parsed']} products, "
                   f"skipped {self.stats['products_skipped']}, "
                   f"errors {self.stats['errors']}")
    
    def _validate_review(self, review: Dict[str, Any]) -> bool:
        """
        Validate review has required fields.
        
        Args:
            review: Review dictionary
        
        Returns:
            True if valid
        """
        # Check required fields
        required = ['rating', 'text', 'asin', 'parent_asin']
        
        for field in required:
            if field not in review:
                logger.debug(f"Review missing required field: {field}")
                return False
        
        # Check text is not empty
        if not review.get('text') or len(review['text'].strip()) < 10:
            return False
        
        return True
    
    def _validate_product(self, product: Dict[str, Any]) -> bool:
        """
        Validate product has required fields.
        
        Args:
            product: Product dictionary
        
        Returns:
            True if valid
        """
        # Check required fields
        required = ['parent_asin', 'title']
        
        for field in required:
            if field not in product:
                logger.debug(f"Product missing required field: {field}")
                return False
        
        return True
    
    def _clean_review(self, review: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and extract relevant review fields.
        
        Args:
            review: Raw review dictionary
        
        Returns:
            Cleaned review dictionary
        """
        # Calculate text length
        text = review.get('text', '')
        text_length = len(text.split()) if text else 0
        
        # Check if has images
        has_images = bool(review.get('images'))
        
        return {
            'parent_asin': review['parent_asin'],
            'user_id': review.get('user_id'),
            'rating': float(review['rating']),
            'title': review.get('title', ''),
            'text': text,
            'text_length': text_length,
            'timestamp': review.get('timestamp'),
            'verified_purchase': review.get('verified_purchase', False),
            'helpful_vote': review.get('helpful_vote', 0),
            'has_images': has_images
        }
    
    def _clean_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and extract relevant product fields.
        
        Args:
            product: Raw product dictionary
        
        Returns:
            Cleaned product dictionary
        """
        # Extract image URL (first image if available)
        image_url = None
        if product.get('images') and len(product['images']) > 0:
            first_image = product['images'][0]
            image_url = first_image.get('large') or first_image.get('hi_res')
        
        # Convert features and description to JSON strings
        features = json.dumps(product.get('features', []))
        description = json.dumps(product.get('description', []))
        
        # Store full metadata as JSON for future use
        product_metadata = json.dumps(product)
        
        return {
            'parent_asin': product['parent_asin'],
            'title': product['title'],
            'average_rating': product.get('average_rating'),
            'rating_number': product.get('rating_number', 0),
            'price': product.get('price'),
            'image_url': image_url,
            'features': features,
            'description': description,
            'product_metadata': product_metadata,
            'details': product.get('details', {}),  # For brand extraction
            'store': product.get('store'),  # For brand extraction
            'main_category': product.get('main_category')
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get parsing statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset parsing statistics."""
        self.stats = {
            'reviews_parsed': 0,
            'reviews_skipped': 0,
            'products_parsed': 0,
            'products_skipped': 0,
            'errors': 0
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    parser = DataParser()
    
    # Test with sample data (if exists)
    sample_review_file = "./data/raw/electronics/Electronics.jsonl.gz"
    sample_meta_file = "./data/raw/electronics/meta_Electronics.jsonl.gz"
    
    if Path(sample_review_file).exists():
        print("\n📖 Testing review parsing (first 5):")
        print("=" * 60)
        
        for i, review in enumerate(parser.parse_reviews(sample_review_file, limit=5), 1):
            print(f"\nReview {i}:")
            print(f"  ASIN: {review['parent_asin']}")
            print(f"  Rating: {review['rating']} ⭐")
            print(f"  Text: {review['text'][:100]}...")
            print(f"  Length: {review['text_length']} words")
        
        print("\n" + "=" * 60)
        print(f"Stats: {parser.get_stats()}")
    else:
        print(f"⚠️  Sample file not found: {sample_review_file}")
        print("   Run: python scripts/download_data.py --category electronics")
    
    parser.reset_stats()
    
    if Path(sample_meta_file).exists():
        print("\n📦 Testing metadata parsing (first 5):")
        print("=" * 60)
        
        for i, product in enumerate(parser.parse_metadata(sample_meta_file, limit=5), 1):
            print(f"\nProduct {i}:")
            print(f"  ASIN: {product['parent_asin']}")
            print(f"  Title: {product['title'][:60]}...")
            print(f"  Rating: {product.get('average_rating', 'N/A')}")
            print(f"  Reviews: {product.get('rating_number', 0)}")
        
        print("\n" + "=" * 60)
        print(f"Stats: {parser.get_stats()}")
    else:
        print(f"⚠️  Sample file not found: {sample_meta_file}")

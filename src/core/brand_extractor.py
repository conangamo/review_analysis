"""Brand extraction and normalization utilities."""

import re
from typing import Optional, Dict, Any, List

try:
    from .config_loader import ConfigLoader
except ImportError:
    # For direct execution
    from config_loader import ConfigLoader


class BrandExtractor:
    """Extract and normalize brand names from product data."""
    
    def __init__(self, category_name: str, config_loader: Optional[ConfigLoader] = None):
        """
        Initialize brand extractor.
        
        Args:
            category_name: Category name
            config_loader: Optional config loader instance
        """
        if config_loader is None:
            config_loader = ConfigLoader()
        
        self.category_name = category_name
        self.config = config_loader.get_brand_config(category_name)
        
        # Build brand normalization map
        self.brand_map = self._build_brand_map()
        self.blacklist = set(word.lower() for word in self.config.get('blacklist', []))
    
    def _build_brand_map(self) -> Dict[str, str]:
        """Build brand normalization mapping."""
        brand_map = {}
        
        for brand_info in self.config.get('known_brands', []):
            canonical = brand_info['canonical']
            
            # Map canonical to itself
            brand_map[canonical.lower()] = canonical
            
            # Map all variations to canonical
            for variation in brand_info.get('variations', []):
                brand_map[variation.lower()] = canonical
        
        return brand_map
    
    def extract_brand(self, product_data: Dict[str, Any]) -> str:
        """
        Extract brand from product data.
        
        Priority:
        1. metadata.details.Brand
        2. metadata.store (if available)
        3. First word in title (fallback)
        4. "Unknown" (if all else fails)
        
        Args:
            product_data: Product data dictionary
        
        Returns:
            Normalized brand name
        """
        # Priority 1: Look in details.Brand
        if 'details' in product_data:
            brand = product_data['details'].get('Brand')
            if brand:
                return self.normalize_brand(brand)
        
        # Priority 2: Look in store field
        if 'store' in product_data and product_data['store']:
            return self.normalize_brand(product_data['store'])
        
        # Priority 3: Extract from title
        if 'title' in product_data:
            brand = self._extract_from_title(product_data['title'])
            if brand:
                return self.normalize_brand(brand)
        
        # Priority 4: Unknown
        return "Unknown"
    
    def _extract_from_title(self, title: str) -> Optional[str]:
        """
        Extract brand from product title.
        
        Args:
            title: Product title
        
        Returns:
            Extracted brand name or None
        """
        if not title:
            return None
        
        # Clean and split title
        words = title.strip().split()
        if not words:
            return None
        
        # Try first word
        first_word = re.sub(r'[^a-zA-Z0-9]', '', words[0])
        
        if first_word and first_word.lower() not in self.blacklist:
            return first_word
        
        # Try second word if first is blacklisted
        if len(words) > 1:
            second_word = re.sub(r'[^a-zA-Z0-9]', '', words[1])
            if second_word and second_word.lower() not in self.blacklist:
                return second_word
        
        return None
    
    def normalize_brand(self, brand: str) -> str:
        """
        Normalize brand name using known brands map.
        
        Args:
            brand: Raw brand name
        
        Returns:
            Normalized brand name
        """
        if not brand:
            return "Unknown"
        
        # Clean brand name
        brand_clean = brand.strip()
        brand_lower = brand_clean.lower()
        
        # Check if in known brands map
        if brand_lower in self.brand_map:
            return self.brand_map[brand_lower]
        
        # Return title-cased version for unknown brands
        return brand_clean.title()
    
    def is_valid_brand(self, brand: str) -> bool:
        """
        Check if brand name is valid (not Unknown or blacklisted).
        
        Args:
            brand: Brand name
        
        Returns:
            True if valid brand
        """
        if not brand or brand == "Unknown":
            return False
        
        if brand.lower() in self.blacklist:
            return False
        
        return True
    
    def get_brand_variations(self, canonical_brand: str) -> List[str]:
        """
        Get all known variations of a canonical brand name.
        
        Args:
            canonical_brand: Canonical brand name
        
        Returns:
            List of all variations
        """
        variations = [canonical_brand]
        
        for brand_info in self.config.get('known_brands', []):
            if brand_info['canonical'] == canonical_brand:
                variations.extend(brand_info.get('variations', []))
                break
        
        return variations
    
    def suggest_brand_addition(self, unknown_brand: str, count: int = 0):
        """
        Suggest adding an unknown brand to the known brands list.
        
        Args:
            unknown_brand: Brand name that should be added
            count: Number of products with this brand
        """
        print(f"\n💡 Suggestion: Add '{unknown_brand}' to known_brands")
        print(f"   Products with this brand: {count}")
        print(f"   Add to config/categories/{self.category_name}.yaml:")
        print(f"""
    - canonical: "{unknown_brand}"
      variations: ["{unknown_brand.upper()}", "{unknown_brand.lower()}"]
        """)


# Example usage
if __name__ == "__main__":
    # Test brand extraction
    extractor = BrandExtractor("electronics")
    
    # Test cases
    test_products = [
        {
            'title': 'Sony WH-1000XM4 Wireless Headphones',
            'details': {'Brand': 'Sony'}
        },
        {
            'title': 'Samsung Galaxy S21 Phone',
            'details': {}
        },
        {
            'title': 'Wireless Bluetooth Speaker',
            'details': {},
            'store': 'Anker'
        },
        {
            'title': 'USB-C Cable 6ft',
            'details': {}
        }
    ]
    
    print("Brand Extraction Tests:")
    print("=" * 50)
    
    for product in test_products:
        brand = extractor.extract_brand(product)
        print(f"Title: {product['title'][:40]}...")
        print(f"  → Brand: {brand}")
        print(f"  → Valid: {extractor.is_valid_brand(brand)}")
        print()

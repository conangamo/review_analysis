"""Product sampling strategies for balanced datasets."""

import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class ProductSampler:
    """Sample products using various strategies."""
    
    def __init__(self, min_reviews: int = 20):
        """
        Initialize sampler.
        
        Args:
            min_reviews: Minimum number of reviews per product
        """
        self.min_reviews = min_reviews
        self.stats = {
            'total_products': 0,
            'filtered_products': 0,
            'selected_products': 0,
            'by_bin': {}
        }
    
    def stratified_sample(
        self,
        products: List[Dict[str, Any]],
        rating_bins: List[Tuple[float, float, int]]
    ) -> List[str]:
        """
        Select products using stratified sampling.
        
        Strategy:
        - Filter products by min_reviews
        - Group by rating bins
        - Select top N from each bin (sorted by review count)
        
        Args:
            products: List of product dictionaries
            rating_bins: List of (min_rating, max_rating, sample_count) tuples
        
        Returns:
            List of selected parent_asins
        """
        logger.info(f"Starting stratified sampling...")
        logger.info(f"Total products: {len(products)}")
        logger.info(f"Min reviews threshold: {self.min_reviews}")
        
        self.stats['total_products'] = len(products)
        
        # Step 1: Filter by minimum reviews
        filtered = self._filter_by_reviews(products)
        logger.info(f"After filtering (≥{self.min_reviews} reviews): {len(filtered)}")
        
        self.stats['filtered_products'] = len(filtered)
        
        # Step 2: Group by rating bins
        bins = self._group_by_rating(filtered, rating_bins)
        
        # Step 3: Sample from each bin
        selected_asins = []
        
        for (min_rating, max_rating, sample_count), products_in_bin in bins.items():
            # Sort by review count (descending)
            sorted_products = sorted(
                products_in_bin,
                key=lambda p: p.get('rating_number', 0),
                reverse=True
            )
            
            # Take top N
            sampled = sorted_products[:sample_count]
            sampled_asins = [p['parent_asin'] for p in sampled]
            
            selected_asins.extend(sampled_asins)
            
            # Log stats
            bin_name = f"{min_rating}-{max_rating}"
            self.stats['by_bin'][bin_name] = {
                'available': len(products_in_bin),
                'selected': len(sampled),
                'requested': sample_count
            }
            
            logger.info(
                f"Bin [{min_rating}-{max_rating}]: "
                f"Selected {len(sampled)}/{sample_count} "
                f"(available: {len(products_in_bin)})"
            )
        
        self.stats['selected_products'] = len(selected_asins)
        
        logger.info(f"Total selected: {len(selected_asins)} products")
        
        return selected_asins
    
    def top_n_sample(
        self,
        products: List[Dict[str, Any]],
        n: int = 3000
    ) -> List[str]:
        """
        Simple top-N sampling by review count.
        
        Args:
            products: List of product dictionaries
            n: Number of products to select
        
        Returns:
            List of selected parent_asins
        """
        logger.info(f"Starting top-{n} sampling...")
        
        # Filter by minimum reviews
        filtered = self._filter_by_reviews(products)
        
        # Sort by review count
        sorted_products = sorted(
            filtered,
            key=lambda p: p.get('rating_number', 0),
            reverse=True
        )
        
        # Take top N
        selected = sorted_products[:n]
        selected_asins = [p['parent_asin'] for p in selected]
        
        logger.info(f"Selected {len(selected_asins)} products")
        
        return selected_asins
    
    def _filter_by_reviews(
        self,
        products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter products by minimum review count.
        
        Args:
            products: List of product dictionaries
        
        Returns:
            Filtered list
        """
        filtered = [
            p for p in products
            if p.get('rating_number', 0) >= self.min_reviews
        ]
        
        return filtered
    
    def _group_by_rating(
        self,
        products: List[Dict[str, Any]],
        rating_bins: List[Tuple[float, float, int]]
    ) -> Dict[Tuple[float, float, int], List[Dict[str, Any]]]:
        """
        Group products by rating bins.
        
        Args:
            products: List of product dictionaries
            rating_bins: List of (min_rating, max_rating, sample_count) tuples
        
        Returns:
            Dictionary mapping bins to products
        """
        bins = defaultdict(list)
        
        for product in products:
            rating = product.get('average_rating')
            
            if rating is None:
                continue
            
            # Find matching bin
            for bin_spec in rating_bins:
                min_rating, max_rating, _ = bin_spec
                
                if min_rating <= rating <= max_rating:
                    bins[bin_spec].append(product)
                    break
        
        return dict(bins)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sampling statistics."""
        return self.stats.copy()
    
    def print_stats(self):
        """Print sampling statistics."""
        print("\n" + "="*60)
        print("📊 Sampling Statistics")
        print("="*60)
        print(f"Total products: {self.stats['total_products']:,}")
        print(f"After filtering (≥{self.min_reviews} reviews): {self.stats['filtered_products']:,}")
        print(f"Selected: {self.stats['selected_products']:,}")
        
        if self.stats['by_bin']:
            print("\nBy Rating Bin:")
            for bin_name, bin_stats in self.stats['by_bin'].items():
                print(f"  {bin_name}: {bin_stats['selected']}/{bin_stats['requested']} "
                      f"(available: {bin_stats['available']})")
        
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Sample data for testing
    sample_products = [
        {'parent_asin': 'B001', 'average_rating': 4.8, 'rating_number': 1000},
        {'parent_asin': 'B002', 'average_rating': 4.5, 'rating_number': 500},
        {'parent_asin': 'B003', 'average_rating': 3.5, 'rating_number': 300},
        {'parent_asin': 'B004', 'average_rating': 2.5, 'rating_number': 200},
        {'parent_asin': 'B005', 'average_rating': 4.9, 'rating_number': 2000},
        {'parent_asin': 'B006', 'average_rating': 3.0, 'rating_number': 100},
        {'parent_asin': 'B007', 'average_rating': 1.5, 'rating_number': 50},
        {'parent_asin': 'B008', 'average_rating': 4.7, 'rating_number': 1500},
        {'parent_asin': 'B009', 'average_rating': 3.8, 'rating_number': 400},
        {'parent_asin': 'B010', 'average_rating': 2.0, 'rating_number': 25},  # Will be filtered out
    ]
    
    # Test stratified sampling
    print("Testing Stratified Sampling:")
    print("="*60)
    
    sampler = ProductSampler(min_reviews=30)
    
    rating_bins = [
        (4.5, 5.0, 3),  # High-rated: select 3
        (3.0, 4.5, 3),  # Mid-rated: select 3
        (1.0, 3.0, 2),  # Low-rated: select 2
    ]
    
    selected = sampler.stratified_sample(sample_products, rating_bins)
    
    print(f"\nSelected ASINs: {selected}")
    
    sampler.print_stats()
    
    # Test top-N sampling
    print("\nTesting Top-N Sampling:")
    print("="*60)
    
    sampler2 = ProductSampler(min_reviews=30)
    selected2 = sampler2.top_n_sample(sample_products, n=5)
    
    print(f"\nSelected ASINs: {selected2}")

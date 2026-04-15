"""Sentiment analyzer using aspect-based analysis."""

import hashlib
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from ..core.aspect_manager import AspectManager
from .negation_handler import NegationHandler  # 🔧 SPRINT 2: Add negation handling

if TYPE_CHECKING:
    from .models.zero_shot import ZeroShotClassifier

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze sentiment for product reviews with aspect detection."""
    
    def __init__(
        self,
        category_name: str,
        aspect_manager: AspectManager,
        classifier: Optional["ZeroShotClassifier"] = None,
        use_keyword_filter: bool = True,
        confidence_threshold: float = 0.65,  # 🔧 FIX: Raised from 0.3 to 0.65
        min_confidence_tier1: float = 0.55,   # 🔧 NEW: Lower threshold for tier 1 (core aspects)
        min_confidence_tier2: float = 0.70,   # 🔧 NEW: Higher threshold for tier 2/3
        strict_keyword_mode: bool = True,     # 🔧 NEW: Pass to aspect_manager
        use_negation_handling: bool = True    # 🔧 SPRINT 2: Enable negation handling
    ):
        """
        Initialize sentiment analyzer.
        
        Args:
            category_name: Category name
            aspect_manager: Aspect manager instance
            classifier: Optional classifier instance
            use_keyword_filter: Use keyword pre-filtering
            confidence_threshold: Default minimum confidence (0.65 recommended)
            min_confidence_tier1: Minimum confidence for tier 1 aspects (0.55 recommended)
            min_confidence_tier2: Minimum confidence for tier 2/3 aspects (0.70 recommended)
            strict_keyword_mode: Require keyword matching for all aspects
            use_negation_handling: Enable negation and contrast detection (Sprint 2)
        """
        self.category_name = category_name
        self.aspect_manager = aspect_manager
        self.use_keyword_filter = use_keyword_filter
        self.confidence_threshold = confidence_threshold
        self.min_confidence_tier1 = min_confidence_tier1
        self.min_confidence_tier2 = min_confidence_tier2
        self.strict_keyword_mode = strict_keyword_mode
        self.use_negation_handling = use_negation_handling
        
        # Initialize classifier if not provided
        if classifier is None:
            from .models.zero_shot import ZeroShotClassifier
            classifier = ZeroShotClassifier()
        self.classifier = classifier
        
        # 🔧 SPRINT 2: Initialize negation handler
        self.negation_handler = NegationHandler() if use_negation_handling else None
        
        # Cache for results
        self.cache = {}
    
    def analyze_review(
        self,
        review_text: str,
        review_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze a single review for aspect-based sentiment.
        
        Args:
            review_text: Review text
            review_id: Optional review ID for caching
        
        Returns:
            List of aspect sentiment results
        """
        if not review_text or len(review_text.strip()) < 10:
            return []
        
        # Check cache
        cache_key = self._get_cache_key(review_text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Step 1: Determine which aspects to analyze
        aspects_to_analyze = self._get_aspects_to_analyze(review_text)
        
        if not aspects_to_analyze:
            logger.debug(f"No aspects detected in review: {review_text[:50]}...")
            return []
        
        # Step 2: Analyze sentiment for each aspect
        results = []
        
        for aspect in aspects_to_analyze:
            aspect_name = aspect['name']
            aspect_tier = aspect['tier']
            aspect_keywords = aspect.get('keywords', [])
            
            # 🔧 SPRINT 2: Process with negation handling
            text_to_analyze = review_text
            has_contrast = False
            
            if self.use_negation_handling and self.negation_handler:
                # Check for contrast words and extract relevant clause
                contrast_result = self.negation_handler.process_review_with_contrast(
                    review_text, aspect_keywords
                )
                
                if contrast_result['has_contrast']:
                    text_to_analyze = contrast_result['text_to_analyze']
                    has_contrast = True
            
            # Classify sentiment
            sentiment_result = self.classifier.classify_sentiment(
                text_to_analyze,
                aspect_name
            )
            
            # 🔧 SPRINT 2: Adjust for negation if present
            if self.use_negation_handling and self.negation_handler:
                negation_result = self.negation_handler.analyze_sentiment_with_negation(
                    text_to_analyze,
                    sentiment_result['sentiment'],
                    sentiment_result['confidence']
                )
                
                # Use adjusted sentiment if modification was made
                if negation_result['adjustment_made']:
                    sentiment_result['sentiment'] = negation_result['sentiment']
                    sentiment_result['confidence'] = negation_result['confidence']
                    sentiment_result['negation_adjusted'] = True
                    sentiment_result['original_sentiment'] = negation_result['original_sentiment']
            
            # 🔧 FIX: Use tiered confidence thresholds
            # Tier 1 (core) aspects have lower threshold, tier 2/3 need higher confidence
            if aspect_tier == 1:
                min_conf = self.min_confidence_tier1
            else:
                min_conf = self.min_confidence_tier2
            
            # Only include if confidence is above the appropriate threshold
            if sentiment_result['confidence'] >= min_conf:
                result_dict = {
                    'aspect': aspect_name,
                    'tier': aspect_tier,
                    'sentiment': sentiment_result['sentiment'],
                    'confidence': sentiment_result['confidence'],
                    'detection_method': 'keyword+ml' if self.use_keyword_filter else 'ml'
                }
                
                # 🔧 SPRINT 2: Add negation metadata
                if self.use_negation_handling:
                    result_dict['has_contrast'] = has_contrast
                    result_dict['negation_adjusted'] = sentiment_result.get('negation_adjusted', False)
                    if result_dict['negation_adjusted']:
                        result_dict['original_sentiment'] = sentiment_result.get('original_sentiment')
                
                results.append(result_dict)
        
        # Cache results
        self.cache[cache_key] = results
        
        return results
    
    def _get_aspects_to_analyze(self, text: str) -> List[Dict[str, Any]]:
        """
        Determine which aspects to analyze for given text.
        
        Args:
            text: Review text
        
        Returns:
            List of aspects to analyze
        """
        if self.use_keyword_filter:
            # 🔧 FIX: Use strict keyword matching (no tier 1 hallucination)
            return self.aspect_manager.get_aspects_for_analysis(
                text,
                include_tier1_always=False,           # Don't force tier 1
                strict_keyword_matching=self.strict_keyword_mode
            )
        else:
            # Analyze all aspects (slower but more comprehensive)
            return self.aspect_manager.aspects
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for review text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.cache.clear()
        logger.info("Analysis cache cleared")
    
    def get_cache_size(self) -> int:
        """Get number of cached results."""
        return len(self.cache)
    
    def get_overall_sentiment(self, review_text: str) -> Dict[str, Any]:
        """
        Get overall sentiment without aspect breakdown.
        
        Args:
            review_text: Review text
        
        Returns:
            Overall sentiment result
        """
        result = self.classifier.classify(
            review_text,
            ["positive", "negative", "neutral"],
            multi_label=False,
            hypothesis_template="This review is {}."
        )
        
        return {
            'sentiment': result['labels'][0],
            'confidence': result['scores'][0],
            'all_scores': {
                label: score 
                for label, score in zip(result['labels'], result['scores'])
            }
        }
    
    def analyze_reviews_batch(
        self,
        reviews: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple reviews in batch.
        
        Args:
            reviews: List of review dictionaries with 'id' and 'text' keys
            show_progress: Show progress bar
        
        Returns:
            List of analysis results
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(reviews, desc="Analyzing reviews")
            except ImportError:
                iterator = reviews
        else:
            iterator = reviews
        
        for review in iterator:
            review_id = review.get('id')
            review_text = review.get('text', '')
            
            aspect_sentiments = self.analyze_review(review_text, review_id)
            
            results.append({
                'review_id': review_id,
                'aspects': aspect_sentiments
            })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            'category': self.category_name,
            'total_aspects': len(self.aspect_manager.aspects),
            'tier_1_aspects': len(self.aspect_manager.get_aspects_by_tier(1)),
            'tier_2_aspects': len(self.aspect_manager.get_aspects_by_tier(2)),
            'tier_3_aspects': len(self.aspect_manager.get_aspects_by_tier(3)),
            'use_keyword_filter': self.use_keyword_filter,
            'confidence_threshold': self.confidence_threshold,
            'cache_size': self.get_cache_size(),
            'model_info': self.classifier.get_model_info()
        }


# Example usage
if __name__ == "__main__":
    from ..core.aspect_manager import AspectManager
    
    # Initialize
    aspect_manager = AspectManager("electronics")
    analyzer = SentimentAnalyzer("electronics", aspect_manager)
    
    # Test review
    review = """
    This phone is amazing! The battery lasts all day even with heavy use.
    The screen is bright and vibrant, perfect for watching videos.
    However, the price is a bit steep for what you get. Camera quality
    is just okay, nothing special. Overall, good value if you can find
    it on sale.
    """
    
    print("Analyzing review:")
    print(f"{review}\n")
    print("="*60)
    
    # Analyze
    results = analyzer.analyze_review(review)
    
    print(f"\nFound {len(results)} aspects:\n")
    for result in results:
        print(f"Aspect: {result['aspect']:15s} | "
              f"Sentiment: {result['sentiment']:8s} | "
              f"Confidence: {result['confidence']:.3f} | "
              f"Tier: {result['tier']}")
    
    print("\n" + "="*60)
    print("\nAnalyzer Stats:")
    stats = analyzer.get_stats()
    for key, value in stats.items():
        if key != 'model_info':
            print(f"  {key}: {value}")

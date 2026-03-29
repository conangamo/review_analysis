"""Entity extraction for product model detection and validation."""

import re
from typing import List, Dict, Any, Optional, Set, Tuple


class EntityExtractor:
    """Extract and match entities (product models, brands) from review text."""
    
    # Common product model patterns
    MODEL_PATTERNS = [
        # Format: Letter + Number (G1, G5, FT7, etc.)
        r'\b([A-Z]+\d+[A-Z]*)\b',
        
        # Format: Number + Letter (310XT, 920XT)
        r'\b(\d+[A-Z]+)\b',
        
        # Format: Model names with spaces (Edge 530, Fenix 5)
        r'\b(Edge|Fenix|Forerunner|Vivoactive|Instinct)\s+(\d+[A-Za-z]*)\b',
        
        # Format: Full model codes (FR235, FR945)
        r'\b(FR\d+)\b',
    ]
    
    # Common entity types
    ENTITY_TYPES = {
        'product_model': MODEL_PATTERNS,
        'brand': [r'\b(Garmin|Polar|Suunto|Apple|Samsung|Fitbit|TomTom)\b'],
        'competitor': [r'\b(competitor|alternative|instead|rather than|versus|vs)\b']
    }
    
    def __init__(self):
        """Initialize entity extractor."""
        self.compiled_patterns = {
            entity_type: [re.compile(pattern, re.IGNORECASE) 
                         for pattern in patterns]
            for entity_type, patterns in self.ENTITY_TYPES.items()
        }
    
    def extract_product_models(self, text: str) -> List[str]:
        """
        Extract product model mentions from text.
        
        Args:
            text: Review text
        
        Returns:
            List of extracted model names
        """
        models = set()
        
        for pattern in self.compiled_patterns['product_model']:
            matches = pattern.findall(text)
            
            for match in matches:
                if isinstance(match, tuple):
                    # Handle patterns with groups
                    model = ' '.join(m for m in match if m)
                else:
                    model = match
                
                # Filter out common false positives
                if self._is_valid_model(model):
                    models.add(model.upper())
        
        return sorted(list(models))
    
    def _is_valid_model(self, model: str) -> bool:
        """Check if extracted string is likely a valid product model."""
        # Filter out common false positives
        false_positives = {
            'AM', 'PM', 'USA', 'GPS', 'ANT', 'USB', 'LCD', 'LED',
            'HD', 'SD', 'XL', 'XXL', 'UK', 'EU', 'US', 'AA', 'AAA'
        }
        
        model_upper = model.upper()
        
        # Too short (single letter)
        if len(model) <= 1:
            return False
        
        # Known false positive
        if model_upper in false_positives:
            return False
        
        # Should have at least one digit for most product models
        if not any(c.isdigit() for c in model):
            return False
        
        return True
    
    def extract_brands(self, text: str) -> List[str]:
        """
        Extract brand mentions from text.
        
        Args:
            text: Review text
        
        Returns:
            List of brand names
        """
        brands = set()
        
        for pattern in self.compiled_patterns['brand']:
            matches = pattern.findall(text)
            for match in matches:
                brands.add(match.title())  # Capitalize properly
        
        return sorted(list(brands))
    
    def detect_competitor_mention(self, text: str) -> bool:
        """
        Check if review mentions competitors or alternatives.
        
        Args:
            text: Review text
        
        Returns:
            True if competitor mention detected
        """
        for pattern in self.compiled_patterns['competitor']:
            if pattern.search(text):
                return True
        return False
    
    def validate_review_entity(
        self,
        review_text: str,
        expected_model: Optional[str] = None,
        expected_brand: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate if review mentions correct product.
        
        Args:
            review_text: Review text
            expected_model: Expected product model
            expected_brand: Expected brand
        
        Returns:
            Validation result:
            {
                'is_valid': bool,
                'mentioned_models': List[str],
                'mentioned_brands': List[str],
                'has_mismatch': bool,
                'mismatch_type': str,  # 'model', 'brand', 'competitor', 'none'
                'confidence': float,
                'warning': str
            }
        """
        mentioned_models = self.extract_product_models(review_text)
        mentioned_brands = self.extract_brands(review_text)
        has_competitor = self.detect_competitor_mention(review_text)
        
        result = {
            'is_valid': True,
            'mentioned_models': mentioned_models,
            'mentioned_brands': mentioned_brands,
            'has_mismatch': False,
            'mismatch_type': 'none',
            'confidence': 1.0,
            'warning': None
        }
        
        # Check model mismatch
        if expected_model and mentioned_models:
            expected_upper = expected_model.upper()
            
            # Normalize expected model for comparison
            expected_normalized = self._normalize_model_name(expected_upper)
            
            # Check if any mentioned model matches expected
            has_match = False
            for model in mentioned_models:
                model_normalized = self._normalize_model_name(model)
                if expected_normalized in model_normalized or model_normalized in expected_normalized:
                    has_match = True
                    break
            
            if not has_match:
                result['is_valid'] = False
                result['has_mismatch'] = True
                result['mismatch_type'] = 'model'
                result['confidence'] = 0.3
                result['warning'] = f"Review mentions {', '.join(mentioned_models)} but product is {expected_model}"
        
        # Check brand mismatch
        if expected_brand and mentioned_brands:
            expected_brand_lower = expected_brand.lower()
            mentioned_brands_lower = [b.lower() for b in mentioned_brands]
            
            if expected_brand_lower not in mentioned_brands_lower:
                # Check if it's a competitor brand
                if len(mentioned_brands) > 0:
                    result['is_valid'] = False
                    result['has_mismatch'] = True
                    result['mismatch_type'] = 'brand'
                    result['confidence'] = 0.4
                    result['warning'] = f"Review mentions {', '.join(mentioned_brands)} but product is {expected_brand}"
        
        # Check competitor mention
        if has_competitor and (mentioned_models or mentioned_brands):
            if not result['has_mismatch']:  # Don't override existing mismatch
                result['mismatch_type'] = 'competitor'
                result['confidence'] = 0.7  # Less severe than direct mismatch
                result['warning'] = "Review mentions competitor products"
        
        return result
    
    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for comparison."""
        # Remove spaces, hyphens
        normalized = model.upper().replace(' ', '').replace('-', '')
        
        # Common abbreviations
        normalized = normalized.replace('FORERUNNER', 'FR')
        normalized = normalized.replace('VIVOACTIVE', 'VA')
        
        return normalized
    
    def extract_context_around_entity(
        self,
        text: str,
        entity: str,
        window_chars: int = 100
    ) -> Optional[str]:
        """
        Extract context around an entity mention.
        
        Args:
            text: Full text
            entity: Entity to find
            window_chars: Characters before/after entity
        
        Returns:
            Context string or None if entity not found
        """
        # Case-insensitive search
        pattern = re.compile(re.escape(entity), re.IGNORECASE)
        match = pattern.search(text)
        
        if not match:
            return None
        
        start = max(0, match.start() - window_chars)
        end = min(len(text), match.end() + window_chars)
        
        context = text[start:end]
        
        # Try to extend to sentence boundaries
        # Find previous sentence end
        prev_period = context.rfind('.', 0, match.start() - start)
        if prev_period != -1:
            context = context[prev_period + 1:]
        
        # Find next sentence end
        next_period = context.find('.', match.end() - start)
        if next_period != -1:
            context = context[:next_period + 1]
        
        return context.strip()
    
    def analyze_review_entities(
        self,
        review_text: str,
        product_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive entity analysis for a review.
        
        Args:
            review_text: Review text
            product_info: Expected product info {'model': str, 'brand': str, 'title': str}
        
        Returns:
            Complete analysis result
        """
        # Extract all entities
        models = self.extract_product_models(review_text)
        brands = self.extract_brands(review_text)
        has_competitor = self.detect_competitor_mention(review_text)
        
        # Validate against expected
        validation = self.validate_review_entity(
            review_text,
            product_info.get('model'),
            product_info.get('brand')
        )
        
        # Get context for mismatched entities
        contexts = {}
        if validation['has_mismatch']:
            for model in models:
                context = self.extract_context_around_entity(review_text, model)
                if context:
                    contexts[model] = context
        
        return {
            'extracted_models': models,
            'extracted_brands': brands,
            'has_competitor_mention': has_competitor,
            'validation': validation,
            'entity_contexts': contexts,
            'recommendation': 'reject' if validation['confidence'] < 0.5 else 'accept_with_warning' if validation['has_mismatch'] else 'accept'
        }


# Example usage and testing
if __name__ == "__main__":
    extractor = EntityExtractor()
    
    print("="*70)
    print("🧪 ENTITY EXTRACTOR TESTS")
    print("="*70)
    
    # Test 1: Model extraction
    print("\n📋 Test 1: Model Extraction")
    print("-"*70)
    
    test_cases = [
        "The G5 is a great little gps unit",
        "I upgraded from 310XT to 920XT",
        "Forerunner 235 works perfectly",
        "Better than my old FR945",
        "Edge 530 vs Fenix 5 comparison"
    ]
    
    for text in test_cases:
        models = extractor.extract_product_models(text)
        print(f"  '{text}'")
        print(f"  → Models: {models}")
        print()
    
    # Test 2: Entity validation (mismatch detection)
    print("📋 Test 2: Entity Mismatch Detection")
    print("-"*70)
    
    # Real case from our data: G5 mentioned in G1 review
    review = "The G5 is a great little gps unit. It's small and fits easily in my cycling jersey"
    expected_model = "G1"
    
    print(f"Review: '{review}'")
    print(f"Expected model: {expected_model}")
    
    validation = extractor.validate_review_entity(review, expected_model=expected_model)
    
    print(f"\nValidation Result:")
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Has mismatch: {validation['has_mismatch']}")
    print(f"  Mismatch type: {validation['mismatch_type']}")
    print(f"  Confidence: {validation['confidence']}")
    print(f"  Warning: {validation['warning']}")
    
    # Test 3: Comprehensive analysis
    print("\n📋 Test 3: Comprehensive Analysis")
    print("-"*70)
    
    product_info = {
        'model': 'Polar G1',
        'brand': 'Polar',
        'title': 'Polar G1 GPS Receiver'
    }
    
    analysis = extractor.analyze_review_entities(review, product_info)
    
    print(f"Analysis Result:")
    print(f"  Models found: {analysis['extracted_models']}")
    print(f"  Brands found: {analysis['extracted_brands']}")
    print(f"  Competitor mention: {analysis['has_competitor_mention']}")
    print(f"  Recommendation: {analysis['recommendation']}")
    
    print("\n" + "="*70)
    print("✅ All tests complete!")

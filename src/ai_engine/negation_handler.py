"""Negation and contrast detection for improved sentiment analysis."""

import re
from typing import List, Dict, Any, Tuple, Optional


class NegationHandler:
    """Handle negation and contrast words in review text."""
    
    # Negation words that reverse sentiment
    NEGATION_WORDS = {
        'not', 'no', 'never', 'neither', 'nor', 'nothing', 'nowhere',
        'nobody', 'none', 'cannot', "can't", "won't", "wouldn't",
        "shouldn't", "couldn't", "doesn't", "don't", "didn't",
        "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
        "hadn't", "without", "lacking", "absence"
    }
    
    # Contrast words that introduce opposing sentiment
    CONTRAST_WORDS = {
        'but', 'however', 'although', 'though', 'yet', 'nevertheless',
        'nonetheless', 'whereas', 'while', 'despite', 'except'
    }
    
    # Intensifiers that modify sentiment strength
    INTENSIFIERS = {
        'very', 'extremely', 'incredibly', 'absolutely', 'totally',
        'completely', 'utterly', 'really', 'quite', 'fairly',
        'pretty', 'somewhat', 'rather'
    }

    # Explicit phrase-level polarity reversals (safer than global flip).
    NEGATE_POSITIVE_PATTERNS = [
        r"\bnot\s+good\b", r"\bnot\s+great\b", r"\bnot\s+worth\b",
        r"\bnot\s+recommended?\b", r"\bnever\s+again\b", r"\bno\s+longer\s+works?\b",
    ]
    NEGATE_NEGATIVE_PATTERNS = [
        r"\bnot\s+bad\b", r"\bnot\s+terrible\b", r"\bnot\s+awful\b",
        r"\bno\s+issues?\b", r"\bno\s+problems?\b",
    ]
    
    def __init__(self):
        """Initialize negation handler."""
        pass
    
    def detect_negation(self, text: str) -> bool:
        """
        Check if text contains negation words.
        
        Args:
            text: Text to check
        
        Returns:
            True if negation detected
        """
        text_lower = text.lower()
        words = set(text_lower.split())
        
        return bool(words.intersection(self.NEGATION_WORDS))
    
    def detect_contrast(self, text: str) -> bool:
        """
        Check if text contains contrast words.
        
        Args:
            text: Text to check
        
        Returns:
            True if contrast detected
        """
        text_lower = text.lower()
        
        for word in self.CONTRAST_WORDS:
            # Match whole word with boundaries
            pattern = r'\b' + word + r'\b'
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def split_on_contrast(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text at contrast words and identify clauses.
        
        Args:
            text: Text to split
        
        Returns:
            List of clauses with metadata:
            [
                {
                    'text': str,
                    'position': 'before'|'after',
                    'contrast_word': str,
                    'has_negation': bool,
                    'priority': int  # Higher = more important
                }
            ]
        """
        clauses = []
        text_lower = text.lower()
        
        # Find all contrast words with their positions
        contrast_matches = []
        for word in self.CONTRAST_WORDS:
            pattern = r'\b' + word + r'\b'
            for match in re.finditer(pattern, text_lower):
                contrast_matches.append({
                    'word': word,
                    'position': match.start(),
                    'end': match.end()
                })
        
        if not contrast_matches:
            # No contrast words - return whole text
            return [{
                'text': text,
                'position': 'whole',
                'contrast_word': None,
                'has_negation': self.detect_negation(text),
                'priority': 1
            }]
        
        # Sort by position (process left to right)
        contrast_matches.sort(key=lambda x: x['position'])
        
        # Split at first contrast word (most common pattern)
        first_contrast = contrast_matches[0]
        split_pos = first_contrast['position']
        
        clause_before = text[:split_pos].strip()
        clause_after = text[first_contrast['end']:].strip()
        
        if clause_before:
            clauses.append({
                'text': clause_before,
                'position': 'before',
                'contrast_word': first_contrast['word'],
                'has_negation': self.detect_negation(clause_before),
                'priority': 1  # Lower priority
            })
        
        if clause_after:
            clauses.append({
                'text': clause_after,
                'position': 'after',
                'contrast_word': first_contrast['word'],
                'has_negation': self.detect_negation(clause_after),
                'priority': 2  # Higher priority (usually author's real opinion)
            })
        
        return clauses
    
    def extract_aspect_context(
        self, 
        text: str, 
        aspect_keywords: List[str],
        context_window: int = 50
    ) -> Optional[str]:
        """
        Extract text around aspect keywords for more accurate sentiment.
        
        Args:
            text: Full review text
            aspect_keywords: Keywords to search for
            context_window: Characters before/after keyword
        
        Returns:
            Context text containing the aspect, or None if not found
        """
        text_lower = text.lower()
        
        for keyword in aspect_keywords:
            keyword_lower = keyword.lower()
            
            # Find keyword position
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            match = re.search(pattern, text_lower)
            
            if match:
                start = max(0, match.start() - context_window)
                end = min(len(text), match.end() + context_window)
                
                # Extract context
                context = text[start:end]
                
                # Try to extend to sentence boundaries
                context = self._extend_to_sentence_boundary(context, text, start, end)
                
                return context
        
        return None
    
    def _extend_to_sentence_boundary(
        self, 
        context: str, 
        full_text: str,
        start: int,
        end: int
    ) -> str:
        """Extend context to include full sentences."""
        # Sentence boundary markers
        boundaries = '.!?'
        
        # Extend left to sentence start
        while start > 0:
            if full_text[start - 1] in boundaries:
                break
            start -= 1
            context = full_text[start] + context
        
        # Extend right to sentence end
        while end < len(full_text):
            context = context + full_text[end]
            if full_text[end] in boundaries:
                break
            end += 1
        
        return context.strip()
    
    def analyze_sentiment_with_negation(
        self,
        text: str,
        base_sentiment: str,
        confidence: float
    ) -> Dict[str, Any]:
        """
        Adjust sentiment based on negation detection.
        
        Args:
            text: Text to analyze
            base_sentiment: Original sentiment (positive/negative/neutral)
            confidence: Original confidence score
        
        Returns:
            Adjusted sentiment result:
            {
                'sentiment': str,
                'confidence': float,
                'has_negation': bool,
                'adjustment_made': bool,
                'original_sentiment': str
            }
        """
        has_negation = self.detect_negation(text)
        
        if not has_negation:
            return {
                'sentiment': base_sentiment,
                'confidence': confidence,
                'has_negation': False,
                'adjustment_made': False,
                'original_sentiment': base_sentiment
            }
        
        # IMPORTANT:
        # Do NOT globally flip polarity whenever negation exists.
        # Only adjust when clear phrase-level cues are present.
        adjusted_sentiment = base_sentiment
        adjustment_made = False
        text_lower = text.lower()

        if base_sentiment == 'positive':
            if any(re.search(p, text_lower) for p in self.NEGATE_POSITIVE_PATTERNS):
                adjusted_sentiment = 'negative'
                adjustment_made = True
        elif base_sentiment == 'negative':
            if any(re.search(p, text_lower) for p in self.NEGATE_NEGATIVE_PATTERNS):
                adjusted_sentiment = 'positive'
                adjustment_made = True

        if adjustment_made:
            confidence = confidence * 0.85
        
        return {
            'sentiment': adjusted_sentiment,
            'confidence': confidence,
            'has_negation': True,
            'adjustment_made': adjustment_made,
            'original_sentiment': base_sentiment
        }
    
    def process_review_with_contrast(
        self,
        text: str,
        aspect_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Process review considering contrast words.
        
        Strategy:
        1. Split on contrast words (but, however)
        2. Determine which clause is most relevant
        3. Return the clause with highest priority
        
        Args:
            text: Review text
            aspect_keywords: Keywords for the aspect being analyzed
        
        Returns:
            Processing result:
            {
                'text_to_analyze': str,
                'full_text': str,
                'has_contrast': bool,
                'selected_clause': str,
                'all_clauses': List[Dict]
            }
        """
        has_contrast = self.detect_contrast(text)
        
        if not has_contrast:
            return {
                'text_to_analyze': text,
                'full_text': text,
                'has_contrast': False,
                'selected_clause': 'whole',
                'all_clauses': [{'text': text, 'priority': 1}]
            }
        
        # Split on contrast
        clauses = self.split_on_contrast(text)
        
        # Find clause that mentions the aspect
        aspect_clause = None
        for clause in clauses:
            clause_lower = clause['text'].lower()
            if any(kw.lower() in clause_lower for kw in aspect_keywords):
                aspect_clause = clause
                break
        
        # If no clause mentions aspect, use highest priority clause
        if aspect_clause is None:
            aspect_clause = max(clauses, key=lambda c: c['priority'])
        
        return {
            'text_to_analyze': aspect_clause['text'],
            'full_text': text,
            'has_contrast': True,
            'selected_clause': aspect_clause['position'],
            'all_clauses': clauses
        }


# Example usage and testing
if __name__ == "__main__":
    handler = NegationHandler()
    
    print("="*70)
    print("🧪 NEGATION HANDLER TESTS")
    print("="*70)
    
    # Test 1: Contrast detection
    print("\n📋 Test 1: Contrast Detection")
    print("-"*70)
    
    test_cases = [
        "Battery is great but screen is dim",
        "Not good, however works fine",
        "Screen is perfect",
        "Although expensive, quality is excellent"
    ]
    
    for text in test_cases:
        has_contrast = handler.detect_contrast(text)
        print(f"  '{text}'")
        print(f"  → Contrast: {has_contrast}")
        print()
    
    # Test 2: Clause splitting
    print("📋 Test 2: Clause Splitting")
    print("-"*70)
    
    text = "Other reviews do not speak well of this device, but my experience is great"
    clauses = handler.split_on_contrast(text)
    
    print(f"Text: '{text}'")
    print(f"\nClauses found: {len(clauses)}")
    for i, clause in enumerate(clauses):
        print(f"\n  Clause {i+1}:")
        print(f"    Text: '{clause['text']}'")
        print(f"    Position: {clause['position']}")
        print(f"    Priority: {clause['priority']}")
        print(f"    Has negation: {clause['has_negation']}")
    
    # Test 3: Sentiment adjustment
    print("\n📋 Test 3: Sentiment Adjustment with Negation")
    print("-"*70)
    
    test_sentiments = [
        ("Battery is not good", "positive", 0.75),
        ("Not bad at all", "negative", 0.70),
        ("Excellent quality", "positive", 0.90)
    ]
    
    for text, sentiment, confidence in test_sentiments:
        result = handler.analyze_sentiment_with_negation(text, sentiment, confidence)
        print(f"\n  Text: '{text}'")
        print(f"  Original: {sentiment} ({confidence:.2f})")
        print(f"  Adjusted: {result['sentiment']} ({result['confidence']:.2f})")
        print(f"  Adjustment made: {result['adjustment_made']}")
    
    print("\n" + "="*70)
    print("✅ All tests complete!")

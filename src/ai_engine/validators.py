"""Validators for detecting anomalies in review analysis."""

import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ReviewValidator:
    """Validate review analysis results and detect anomalies."""
    
    @staticmethod
    def validate_rating_sentiment_consistency(
        rating: float,
        aspect_sentiments: List[Dict[str, Any]],
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Detect if rating and sentiment don't match (possible sarcasm, error, or spam).
        
        Args:
            rating: Review rating (1-5)
            aspect_sentiments: List of aspect sentiment results
            threshold: Threshold for determining mismatch (0.7 = 70% of aspects)
        
        Returns:
            {
                'is_suspicious': bool,
                'mismatch_type': str,  # 'high_rating_negative', 'low_rating_positive', 'none'
                'severity': str,       # 'critical', 'warning', 'info', 'none'
                'confidence': float,
                'recommendation': str,
                'details': dict
            }
        """
        if not aspect_sentiments:
            return {
                'is_suspicious': False,
                'mismatch_type': 'none',
                'severity': 'none',
                'confidence': 0.0,
                'recommendation': 'no_aspects',
                'details': {}
            }
        
        # Calculate sentiment distribution
        positive = sum(1 for a in aspect_sentiments if a['sentiment'] == 'positive')
        negative = sum(1 for a in aspect_sentiments if a['sentiment'] == 'negative')
        neutral = sum(1 for a in aspect_sentiments if a['sentiment'] == 'neutral')
        total = positive + negative + neutral
        
        positive_pct = positive / total if total > 0 else 0
        negative_pct = negative / total if total > 0 else 0
        
        # Classify rating
        is_high_rating = rating >= 4.0
        is_low_rating = rating <= 2.0
        
        # Check for mismatches
        result = {
            'is_suspicious': False,
            'mismatch_type': 'none',
            'severity': 'none',
            'confidence': 0.0,
            'recommendation': 'accept',
            'details': {
                'rating': rating,
                'positive_pct': positive_pct,
                'negative_pct': negative_pct,
                'total_aspects': total
            }
        }
        
        # Case 1: High rating but mostly negative sentiment
        if is_high_rating and negative_pct >= threshold:
            result['is_suspicious'] = True
            result['mismatch_type'] = 'high_rating_negative'
            result['confidence'] = negative_pct
            
            if negative_pct >= 0.9:
                result['severity'] = 'critical'
                result['recommendation'] = 'flag_for_review'
            elif negative_pct >= 0.7:
                result['severity'] = 'warning'
                result['recommendation'] = 'verify'
            else:
                result['severity'] = 'info'
                result['recommendation'] = 'accept_with_note'
        
        # Case 2: Low rating but mostly positive sentiment
        elif is_low_rating and positive_pct >= threshold:
            result['is_suspicious'] = True
            result['mismatch_type'] = 'low_rating_positive'
            result['confidence'] = positive_pct
            
            if positive_pct >= 0.9:
                result['severity'] = 'critical'
                result['recommendation'] = 'flag_for_review'
            elif positive_pct >= 0.7:
                result['severity'] = 'warning'
                result['recommendation'] = 'verify'
            else:
                result['severity'] = 'info'
                result['recommendation'] = 'accept_with_note'
        
        return result
    
    @staticmethod
    def validate_aspect_confidence(
        aspect_sentiments: List[Dict[str, Any]],
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Check if aspect detections have acceptable confidence scores.
        
        Args:
            aspect_sentiments: List of aspect sentiment results
            min_confidence: Minimum acceptable average confidence
        
        Returns:
            Validation result dictionary
        """
        if not aspect_sentiments:
            return {
                'is_valid': True,
                'avg_confidence': 0.0,
                'low_confidence_count': 0,
                'recommendation': 'no_aspects'
            }
        
        confidences = [a['confidence'] for a in aspect_sentiments]
        avg_confidence = sum(confidences) / len(confidences)
        
        low_confidence = [a for a in aspect_sentiments if a['confidence'] < min_confidence]
        low_confidence_count = len(low_confidence)
        low_confidence_pct = low_confidence_count / len(aspect_sentiments)
        
        return {
            'is_valid': avg_confidence >= min_confidence,
            'avg_confidence': avg_confidence,
            'low_confidence_count': low_confidence_count,
            'low_confidence_pct': low_confidence_pct,
            'low_confidence_aspects': [a['aspect'] for a in low_confidence],
            'recommendation': 'accept' if avg_confidence >= min_confidence else 'review'
        }
    
    @staticmethod
    def detect_spam_patterns(
        review_text: str,
        aspect_sentiments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect common spam patterns in reviews.
        
        Args:
            review_text: Review text
            aspect_sentiments: List of aspect sentiment results
        
        Returns:
            Spam detection result
        """
        text_lower = review_text.lower()
        spam_indicators = []
        
        # Pattern 1: Too short (likely not genuine)
        if len(review_text.split()) < 5:
            spam_indicators.append('too_short')
        
        # Pattern 2: All caps
        if review_text.isupper() and len(review_text) > 10:
            spam_indicators.append('all_caps')
        
        # Pattern 3: Excessive punctuation
        punct_count = sum(1 for c in review_text if c in '!?')
        if punct_count > 5:
            spam_indicators.append('excessive_punctuation')
        
        # Pattern 4: Common spam phrases
        spam_phrases = [
            'buy now', 'click here', 'limited time',
            'check out my', 'visit my website'
        ]
        if any(phrase in text_lower for phrase in spam_phrases):
            spam_indicators.append('spam_phrases')
        
        # Pattern 5: No aspects detected (generic review)
        if not aspect_sentiments:
            spam_indicators.append('no_aspects_detected')
        
        is_spam = len(spam_indicators) >= 2  # Require at least 2 indicators
        
        return {
            'is_spam': is_spam,
            'spam_indicators': spam_indicators,
            'spam_score': len(spam_indicators) / 5,  # Normalize to 0-1
            'recommendation': 'reject' if is_spam else 'accept'
        }
    
    @staticmethod
    def validate_review_comprehensive(
        review_text: str,
        rating: float,
        aspect_sentiments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Comprehensive validation combining all checks.
        
        Args:
            review_text: Review text
            rating: Review rating (1-5)
            aspect_sentiments: List of aspect sentiment results
        
        Returns:
            Comprehensive validation result
        """
        # Run all validators
        rating_check = ReviewValidator.validate_rating_sentiment_consistency(
            rating, aspect_sentiments
        )
        
        confidence_check = ReviewValidator.validate_aspect_confidence(
            aspect_sentiments
        )
        
        spam_check = ReviewValidator.detect_spam_patterns(
            review_text, aspect_sentiments
        )
        
        # Determine overall status
        warnings = []
        flags = []
        
        if rating_check['is_suspicious']:
            if rating_check['severity'] == 'critical':
                flags.append(f"Rating-sentiment mismatch: {rating_check['mismatch_type']}")
            else:
                warnings.append(f"Possible {rating_check['mismatch_type']}")
        
        if not confidence_check['is_valid']:
            warnings.append(f"Low average confidence: {confidence_check['avg_confidence']:.2f}")
        
        if spam_check['is_spam']:
            flags.append(f"Spam detected: {', '.join(spam_check['spam_indicators'])}")
        
        # Overall recommendation
        if flags:
            overall_recommendation = 'reject'
            status = 'flagged'
        elif warnings:
            overall_recommendation = 'review'
            status = 'warning'
        else:
            overall_recommendation = 'accept'
            status = 'valid'
        
        return {
            'status': status,
            'overall_recommendation': overall_recommendation,
            'warnings': warnings,
            'flags': flags,
            'rating_validation': rating_check,
            'confidence_validation': confidence_check,
            'spam_detection': spam_check
        }


# Example usage and testing
if __name__ == "__main__":
    # Test case 1: High rating with negative sentiment
    test_aspects_1 = [
        {'aspect': 'battery', 'sentiment': 'negative', 'confidence': 0.8},
        {'aspect': 'performance', 'sentiment': 'negative', 'confidence': 0.9},
        {'aspect': 'quality', 'sentiment': 'negative', 'confidence': 0.7}
    ]
    
    result1 = ReviewValidator.validate_rating_sentiment_consistency(5.0, test_aspects_1)
    print("Test 1 - High rating, negative sentiment:")
    print(f"  Suspicious: {result1['is_suspicious']}")
    print(f"  Type: {result1['mismatch_type']}")
    print(f"  Severity: {result1['severity']}")
    print(f"  Recommendation: {result1['recommendation']}")
    print()
    
    # Test case 2: Comprehensive validation
    review_text = "Did not work, get a garmin...."
    result2 = ReviewValidator.validate_review_comprehensive(
        review_text, 5.0, test_aspects_1
    )
    print("Test 2 - Comprehensive validation:")
    print(f"  Status: {result2['status']}")
    print(f"  Recommendation: {result2['overall_recommendation']}")
    print(f"  Warnings: {result2['warnings']}")
    print(f"  Flags: {result2['flags']}")

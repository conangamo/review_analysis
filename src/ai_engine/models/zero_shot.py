"""Zero-shot classification model wrapper."""

import torch
from transformers import pipeline
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ZeroShotClassifier:
    """Wrapper for zero-shot classification model."""
    
    def __init__(
        self,
        model_name: str = "valhalla/distilbart-mnli-12-3",
        device: Optional[str] = None,
        use_fp16: bool = True,
        batch_size: int = 32,
        max_length: int = 512
    ):
        """
        Initialize zero-shot classifier.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            use_fp16: Use mixed precision (FP16) for faster inference
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = 0 if device == "cuda" else -1
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"FP16: {use_fp16}")
        
        # Load model
        self.pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=self.device,
            framework="pt"
        )
        
        # Enable FP16 if requested and on GPU
        if use_fp16 and device == "cuda":
            try:
                self.pipeline.model.half()
                logger.info("✅ FP16 enabled")
            except Exception as e:
                logger.warning(f"Could not enable FP16: {e}")
        
        # Warm up model
        self._warmup()
    
    def _warmup(self):
        """Warm up model with dummy inference."""
        try:
            _ = self.pipeline(
                "This is a test.",
                ["test"],
                multi_label=False
            )
            logger.info("✅ Model warmed up")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def classify(
        self,
        text: str,
        candidate_labels: List[str],
        multi_label: bool = False,
        hypothesis_template: str = "This text is about {}."
    ) -> Dict[str, Any]:
        """
        Classify text with zero-shot learning.
        
        Args:
            text: Text to classify
            candidate_labels: List of candidate labels
            multi_label: Whether to allow multiple labels
            hypothesis_template: Template for generating hypotheses
        
        Returns:
            Dictionary with 'labels' and 'scores' keys
        """
        if not text or not candidate_labels:
            return {'labels': [], 'scores': []}
        
        try:
            result = self.pipeline(
                text,
                candidate_labels,
                multi_label=multi_label,
                hypothesis_template=hypothesis_template
            )
            return result
        
        except Exception as e:
            logger.error(f"Classification error: {e}")
            # Return neutral result on error
            return {
                'labels': candidate_labels,
                'scores': [1.0 / len(candidate_labels)] * len(candidate_labels)
            }
    
    def classify_aspect_presence(
        self,
        text: str,
        aspect_names: List[str]
    ) -> Dict[str, float]:
        """
        Detect which aspects are mentioned in text.
        
        Args:
            text: Review text
            aspect_names: List of aspect names to check
        
        Returns:
            Dictionary mapping aspect names to confidence scores
        """
        result = self.classify(
            text,
            aspect_names,
            multi_label=True,
            hypothesis_template="This review mentions {}."
        )
        
        return dict(zip(result['labels'], result['scores']))
    
    def classify_sentiment(
        self,
        text: str,
        aspect: str
    ) -> Dict[str, Any]:
        """
        Classify sentiment about a specific aspect.
        
        Args:
            text: Review text
            aspect: Aspect name
        
        Returns:
            Dictionary with 'sentiment' and 'confidence' keys
        """
        labels = [
            f"positive about {aspect}",
            f"negative about {aspect}",
            f"neutral about {aspect}"
        ]
        
        result = self.classify(
            text,
            labels,
            multi_label=False,
            hypothesis_template="{}"
        )
        
        # Extract sentiment from label
        top_label = result['labels'][0]
        sentiment = top_label.split()[0]  # Extract 'positive', 'negative', or 'neutral'
        confidence = result['scores'][0]
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'all_scores': {
                'positive': result['scores'][result['labels'].index(f"positive about {aspect}")],
                'negative': result['scores'][result['labels'].index(f"negative about {aspect}")],
                'neutral': result['scores'][result['labels'].index(f"neutral about {aspect}")]
            }
        }
    
    def batch_classify(
        self,
        texts: List[str],
        candidate_labels: List[str],
        multi_label: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple texts in batch.
        
        Args:
            texts: List of texts
            candidate_labels: List of candidate labels
            multi_label: Whether to allow multiple labels
        
        Returns:
            List of classification results
        """
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            for text in batch:
                result = self.classify(text, candidate_labels, multi_label)
                results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'device': 'cuda' if self.device == 0 else 'cpu',
            'batch_size': self.batch_size,
            'max_length': self.max_length
        }


# Example usage
if __name__ == "__main__":
    # Test the classifier
    classifier = ZeroShotClassifier()
    
    # Test text
    review = "The battery life is excellent, lasting all day. However, the screen is too dim in sunlight."
    
    # Test aspect detection
    print("Testing aspect detection:")
    aspects = ["battery", "screen", "camera", "price"]
    aspect_scores = classifier.classify_aspect_presence(review, aspects)
    
    print(f"Review: {review}\n")
    print("Aspect presence scores:")
    for aspect, score in sorted(aspect_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {aspect:10s}: {score:.3f}")
    
    # Test sentiment classification
    print("\n\nTesting sentiment classification:")
    for aspect in ["battery", "screen"]:
        result = classifier.classify_sentiment(review, aspect)
        print(f"\n{aspect.upper()}:")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Scores: {result['all_scores']}")

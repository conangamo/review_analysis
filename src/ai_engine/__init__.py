"""AI Engine for sentiment analysis."""

from .sentiment_analyzer import SentimentAnalyzer
from .batch_processor import BatchProcessor

try:
    from .models.zero_shot import ZeroShotClassifier
except ModuleNotFoundError:
    ZeroShotClassifier = None

__all__ = ['ZeroShotClassifier', 'SentimentAnalyzer', 'BatchProcessor']

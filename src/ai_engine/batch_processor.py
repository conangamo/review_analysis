"""Optimized batch processor for large-scale analysis."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process reviews in batches with checkpointing and progress tracking."""
    
    def __init__(
        self,
        analyzer,
        batch_size: int = 100,
        checkpoint_dir: str = "./data/cache/checkpoints",
        checkpoint_interval: int = 10
    ):
        """
        Initialize batch processor.
        
        Args:
            analyzer: SentimentAnalyzer instance
            batch_size: Number of reviews per batch
            checkpoint_dir: Directory for checkpoint files
            checkpoint_interval: Save checkpoint every N batches
        """
        self.analyzer = analyzer
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_processed': 0,
            'total_aspects_found': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def process_reviews(
        self,
        reviews: List[Dict[str, Any]],
        callback: Optional[Callable] = None,
        resume_from: Optional[int] = None,
        checkpoint_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process reviews in batches with checkpointing.
        
        Args:
            reviews: List of review dictionaries
            callback: Optional callback function called after each batch
            resume_from: Resume from batch number
            checkpoint_name: Name for checkpoint file
        
        Returns:
            List of processed results
        """
        self.stats['start_time'] = datetime.now()
        
        if checkpoint_name is None:
            checkpoint_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Resume from checkpoint if specified
        start_batch = resume_from if resume_from is not None else 0
        results = self._load_checkpoint(checkpoint_name) if resume_from else []
        
        total_batches = (len(reviews) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Processing {len(reviews)} reviews in {total_batches} batches")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Starting from batch: {start_batch}")
        
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            logger.warning("tqdm not available, progress bar disabled")
        
        # Process in batches
        batch_iterator = range(start_batch, total_batches)
        
        if use_tqdm:
            batch_iterator = tqdm(
                batch_iterator,
                desc="Processing batches",
                initial=start_batch,
                total=total_batches
            )
        
        for batch_num in batch_iterator:
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(reviews))
            
            batch = reviews[start_idx:end_idx]
            
            # Process batch
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
            # Update stats
            self.stats['total_processed'] += len(batch)
            self.stats['total_aspects_found'] += sum(
                len(r['aspects']) for r in batch_results
            )
            
            # Callback
            if callback:
                callback(batch_num, batch_results, self.stats)
            
            # Checkpoint
            if (batch_num + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_name, results, batch_num + 1)
        
        self.stats['end_time'] = datetime.now()
        
        # Final checkpoint
        self._save_checkpoint(checkpoint_name, results, total_batches, final=True)
        
        logger.info(f"✅ Processing complete!")
        self._print_stats()
        
        return results
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a single batch of reviews."""
        results = []
        
        for review in batch:
            try:
                review_id = review.get('id')
                review_text = review.get('text', '')
                
                # Analyze
                aspects = self.analyzer.analyze_review(review_text, review_id)
                
                results.append({
                    'review_id': review_id,
                    'aspects': aspects
                })
            
            except Exception as e:
                logger.error(f"Error processing review {review.get('id')}: {e}")
                self.stats['errors'] += 1
                
                results.append({
                    'review_id': review.get('id'),
                    'aspects': [],
                    'error': str(e)
                })
        
        return results
    
    def _save_checkpoint(
        self,
        name: str,
        results: List[Dict[str, Any]],
        batch_num: int,
        final: bool = False
    ):
        """Save checkpoint to disk."""
        checkpoint_path = self.checkpoint_dir / f"{name}_batch_{batch_num}.pkl"
        
        checkpoint_data = {
            'results': results,
            'batch_num': batch_num,
            'stats': self.stats.copy(),
            'timestamp': datetime.now().isoformat(),
            'final': final
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, name: str) -> List[Dict[str, Any]]:
        """Load most recent checkpoint."""
        # Find latest checkpoint
        checkpoints = list(self.checkpoint_dir.glob(f"{name}_batch_*.pkl"))
        
        if not checkpoints:
            logger.warning(f"No checkpoints found for {name}")
            return []
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        with open(latest, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        logger.info(f"Loaded checkpoint: {latest}")
        logger.info(f"  Batch: {checkpoint_data['batch_num']}")
        logger.info(f"  Results: {len(checkpoint_data['results'])}")
        
        return checkpoint_data['results']
    
    def export_results(
        self,
        results: List[Dict[str, Any]],
        output_file: str,
        format: str = 'json'
    ):
        """
        Export results to file.
        
        Args:
            results: Analysis results
            output_file: Output file path
            format: Output format ('json' or 'csv')
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            import csv
            
            # Flatten results for CSV
            rows = []
            for result in results:
                review_id = result['review_id']
                
                if result['aspects']:
                    for aspect in result['aspects']:
                        rows.append({
                            'review_id': review_id,
                            'aspect': aspect['aspect'],
                            'tier': aspect['tier'],
                            'sentiment': aspect['sentiment'],
                            'confidence': aspect['confidence'],
                            'detection_method': aspect['detection_method']
                        })
                else:
                    rows.append({
                        'review_id': review_id,
                        'aspect': None,
                        'tier': None,
                        'sentiment': None,
                        'confidence': None,
                        'detection_method': None
                    })
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"✅ Results exported to: {output_path}")
    
    def _print_stats(self):
        """Print processing statistics."""
        duration = self.stats['end_time'] - self.stats['start_time']
        
        print("\n" + "="*60)
        print("📊 Batch Processing Statistics")
        print("="*60)
        print(f"Total reviews processed: {self.stats['total_processed']:,}")
        print(f"Total aspects found: {self.stats['total_aspects_found']:,}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Duration: {duration}")
        print(f"Speed: {self.stats['total_processed'] / duration.total_seconds():.2f} reviews/sec")
        print(f"Cache size: {self.analyzer.get_cache_size()}")
        print("="*60 + "\n")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        if stats['start_time'] and stats['end_time']:
            duration = stats['end_time'] - stats['start_time']
            stats['duration_seconds'] = duration.total_seconds()
            stats['reviews_per_second'] = stats['total_processed'] / duration.total_seconds()
        
        return stats


# Example usage
if __name__ == "__main__":
    from ..core.aspect_manager import AspectManager
    from .sentiment_analyzer import SentimentAnalyzer
    
    # Initialize
    aspect_manager = AspectManager("electronics")
    analyzer = SentimentAnalyzer("electronics", aspect_manager)
    processor = BatchProcessor(analyzer, batch_size=10)
    
    # Sample reviews
    sample_reviews = [
        {
            'id': i,
            'text': f"Sample review {i}. Battery is good, screen is bright."
        }
        for i in range(50)
    ]
    
    # Process
    results = processor.process_reviews(sample_reviews)
    
    print(f"\nProcessed {len(results)} reviews")
    print(f"First result: {results[0]}")

"""Run AI sentiment analysis on reviews."""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import ConfigLoader, AspectManager, get_env
from src.database.db_manager import DatabaseManager
from src.database.models import Review, AspectSentiment, ProcessingStatus
from src.ai_engine import ZeroShotClassifier, SentimentAnalyzer, BatchProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _resolve_analysis_settings(config_loader, category_config, env):
    """Resolve analyzer and processor settings from config/env."""
    model_config = config_loader.load_model_config()
    analyzer_config = model_config.get('sentiment_analyzer', {})
    optimization_config = model_config.get('optimization', {})

    category_processing = category_config.get('processing', {})
    cache_dir = env.get_path("CACHE_DIR", "./data/cache")
    checkpoint_dir = env.get_path(
        "CHECKPOINT_DIR",
        str(cache_dir / "checkpoints")
    )

    return {
        'use_keyword_filter': env.get_bool(
            "USE_KEYWORD_FILTER",
            optimization_config.get('use_keyword_filter', True)
        ),
        'confidence_threshold': env.get_float(
            "CONFIDENCE_THRESHOLD",
            analyzer_config.get('confidence_threshold', 0.65)
        ),
        'min_confidence_tier1': env.get_float(
            "MIN_CONFIDENCE_TIER1",
            analyzer_config.get('min_confidence_tier1', 0.55)
        ),
        'min_confidence_tier2': env.get_float(
            "MIN_CONFIDENCE_TIER2",
            analyzer_config.get('min_confidence_tier2', 0.70)
        ),
        'min_confidence_neutral': env.get_float(
            "MIN_CONFIDENCE_NEUTRAL",
            analyzer_config.get('min_confidence_neutral', 0.40)
        ),
        'analysis_batch_size': env.get_int(
            "ANALYSIS_BATCH_SIZE",
            category_processing.get('ai_batch_size', 32)
        ),
        'checkpoint_dir': checkpoint_dir,
    }


def _find_resume_batch(checkpoint_dir: Path, checkpoint_name: str):
    """Find latest checkpoint batch number for a checkpoint name."""
    pattern = f"{checkpoint_name}_batch_*.pkl"
    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None

    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    match = re.search(r"_batch_(\d+)\.pkl$", latest.name)
    if not match:
        return None
    return int(match.group(1))


def create_processing_status(session, category_id: int, stage: str, total_items: int):
    """Create a processing status entry."""
    status = ProcessingStatus(
        category_id=category_id,
        stage=stage,
        status='running',
        total_items=total_items,
        processed_items=0,
        started_at=datetime.now()
    )
    session.add(status)
    session.commit()
    return status.id


def update_processing_status(session, status_id: int, processed: int, status: str = 'running', error: str = None):
    """Update processing status."""
    status_obj = session.get(ProcessingStatus, status_id)
    if status_obj:
        status_obj.processed_items = processed
        status_obj.progress = (processed / status_obj.total_items * 100) if status_obj.total_items > 0 else 0
        status_obj.status = status
        if error:
            status_obj.error_message = error
        if status in ['completed', 'failed']:
            status_obj.completed_at = datetime.now()
        session.commit()


def main():
    """Main analysis function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run AI sentiment analysis on reviews"
    )
    parser.add_argument(
        '--category',
        type=str,
        required=True,
        help='Category name (e.g., electronics)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for AI processing (default: from .env)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of reviews to analyze (for testing)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    parser.add_argument(
        '--resume-batch',
        type=int,
        help='Resume from a specific batch number (overrides auto-detect)'
    )
    parser.add_argument(
        '--checkpoint-name',
        type=str,
        help='Checkpoint name prefix (default: <category>_analysis)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-analysis (skip existing results)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🤖 AI Sentiment Analysis Pipeline")
    print("="*70)
    print(f"Category: {args.category}")
    print()
    
    try:
        # Step 1: Load configuration
        print("📋 Step 1: Loading configuration...")
        print("-" * 70)
        
        config_loader = ConfigLoader()
        env = get_env()
        
        category_config = config_loader.load_category_config(args.category)
        category_name = category_config['category']['name']
        
        print(f"✅ Category: {category_name}")
        
        # Get settings
        batch_size = args.batch_size or env.get_int("BATCH_SIZE", 32)
        use_gpu = env.get_bool("USE_GPU", True)
        use_fp16 = env.get_bool("USE_FP16", True)
        model_name = env.get_str("MODEL_NAME", "valhalla/distilbart-mnli-12-3")
        analysis_settings = _resolve_analysis_settings(config_loader, category_config, env)
        
        print(f"✅ Batch size: {batch_size}")
        print(f"✅ Analysis batch size: {analysis_settings['analysis_batch_size']}")
        print(f"✅ Use GPU: {use_gpu}")
        print(f"✅ Use FP16: {use_fp16}")
        print(f"✅ Model: {model_name}")
        print(f"✅ Confidence threshold: {analysis_settings['confidence_threshold']}")
        print(f"✅ Neutral min confidence: {analysis_settings['min_confidence_neutral']}")
        print()
        
        # Step 2: Initialize database
        print("🗄️  Step 2: Connecting to database...")
        print("-" * 70)
        
        db_path = env.get_path("DB_PATH", "./data/processed/reviews.db")
        db_manager = DatabaseManager(str(db_path))
        
        print(f"✅ Database: {db_path}")
        
        # Get category ID
        with db_manager.get_session() as session:
            from src.database.models import Category
            category_obj = session.query(Category).filter_by(name=category_name).first()
            
            if not category_obj:
                print(f"\n❌ Error: Category '{category_name}' not found in database")
                print(f"   Run: python scripts/parse_data.py --category {args.category}")
                return 1
            
            category_id = category_obj.id
        
        print(f"✅ Category ID: {category_id}")
        print()
        
        # Step 3: Load reviews from database
        print("📖 Step 3: Loading reviews from database...")
        print("-" * 70)
        
        # Load review data (not ORM objects to avoid detached instance issues)
        with db_manager.get_session() as session:
            # Get reviews for selected products only
            from sqlalchemy import select
            from src.database.models import Product as ProductModel
            
            # Get selected product ASINs first
            selected_asins = session.query(ProductModel.parent_asin).filter_by(
                category_id=category_id,
                is_selected=True
            ).all()
            selected_asins = {asin[0] for asin in selected_asins}
            
            if not selected_asins:
                print(f"\n❌ Error: No selected products found")
                return 1
            
            # Get reviews as dictionaries (not ORM objects)
            query = session.query(
                Review.id,
                Review.text
            ).filter(
                Review.parent_asin.in_(selected_asins)
            )
            
            # Skip already analyzed if not force
            if not args.force:
                analyzed_ids = session.query(AspectSentiment.review_id).distinct().all()
                analyzed_ids = {r[0] for r in analyzed_ids}
                
                query = query.filter(Review.id.notin_(analyzed_ids))
            
            # Apply limit if specified
            if args.limit:
                query = query.limit(args.limit)
            
            # Load as list of dicts
            reviews = query.all()
            
            if not reviews:
                print(f"\n❌ Error: No reviews found for selected products")
                print(f"   Make sure you ran: python scripts/load_reviews_only.py --category {args.category}")
                return 1
            
            print(f"✅ Loaded {len(reviews):,} reviews")
        
        print()
        
        # Step 4: Initialize AI components
        print("🤖 Step 4: Initializing AI models...")
        print("-" * 70)
        
        # Aspect manager
        aspect_manager = AspectManager(args.category, config_loader)
        print(f"✅ Aspect manager initialized")
        print(f"   Total aspects: {len(aspect_manager.aspects)}")
        print(f"   Tier 1: {len(aspect_manager.get_aspects_by_tier(1))}")
        print(f"   Tier 2: {len(aspect_manager.get_aspects_by_tier(2))}")
        print(f"   Tier 3: {len(aspect_manager.get_aspects_by_tier(3))}")
        
        # Zero-shot classifier
        device = "cuda" if use_gpu else "cpu"
        classifier = ZeroShotClassifier(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            batch_size=batch_size
        )
        print(f"✅ Classifier loaded")
        
        # Sentiment analyzer
        analyzer = SentimentAnalyzer(
            category_name=args.category,
            aspect_manager=aspect_manager,
            classifier=classifier,
            use_keyword_filter=analysis_settings['use_keyword_filter'],
            confidence_threshold=analysis_settings['confidence_threshold'],
            min_confidence_tier1=analysis_settings['min_confidence_tier1'],
            min_confidence_tier2=analysis_settings['min_confidence_tier2'],
            min_confidence_neutral=analysis_settings['min_confidence_neutral'],
        )
        print(f"✅ Sentiment analyzer initialized")
        
        # Batch processor
        checkpoint_interval = env.get_int("CHECKPOINT_INTERVAL", 10)
        processor = BatchProcessor(
            analyzer=analyzer,
            batch_size=analysis_settings['analysis_batch_size'],
            checkpoint_dir=str(analysis_settings['checkpoint_dir']),
            checkpoint_interval=checkpoint_interval
        )
        print(f"✅ Batch processor initialized")
        print(f"✅ Checkpoint directory: {analysis_settings['checkpoint_dir']}")
        print()
        
        # Step 5: Create processing status
        print("📊 Step 5: Creating processing status...")
        print("-" * 70)
        
        with db_manager.get_session() as session:
            status_id = create_processing_status(
                session,
                category_id,
                'aspect_analysis',
                len(reviews)
            )
        
        print(f"✅ Processing status created (ID: {status_id})")
        print()
        
        # Step 6: Run analysis
        print("🔍 Step 6: Running AI analysis...")
        print("-" * 70)
        print(f"Analyzing {len(reviews):,} reviews...")
        print(f"Batch size: {analysis_settings['analysis_batch_size']}")
        print(f"Checkpoint interval: every {checkpoint_interval} batches")
        print()
        
        # Prepare review data (already in dict format from query)
        review_data = [
            {'id': r[0], 'text': r[1]}
            for r in reviews
        ]
        
        # Callback to save results
        def save_batch_callback(batch_num, batch_results, stats):
            """Save batch results to database."""
            with db_manager.get_session() as session:
                for result in batch_results:
                    review_id = result['review_id']
                    
                    # Skip if error
                    if 'error' in result:
                        continue
                    
                    # 🔧 FIX: Delete existing aspects for this review first (handle --force)
                    if args.force:
                        session.query(AspectSentiment).filter_by(
                            review_id=review_id
                        ).delete()
                    
                    # Save aspect sentiments
                    for aspect_data in result['aspects']:
                        sentiment = AspectSentiment(
                            review_id=review_id,
                            aspect_name=aspect_data['aspect'],
                            aspect_tier=aspect_data['tier'],
                            sentiment=aspect_data['sentiment'],
                            confidence_score=aspect_data['confidence'],
                            detection_method=aspect_data.get('detection_method', 'keyword+ml')
                        )
                        session.add(sentiment)
                
                # Update processing status
                update_processing_status(
                    session,
                    status_id,
                    stats['total_processed']
                )
        
        checkpoint_name = args.checkpoint_name or f"{args.category.lower()}_analysis"
        resume_batch = None
        if args.resume:
            if args.resume_batch is not None:
                resume_batch = args.resume_batch
            else:
                resume_batch = _find_resume_batch(
                    analysis_settings['checkpoint_dir'],
                    checkpoint_name
                )
            print(f"🔄 Resume mode: checkpoint='{checkpoint_name}', batch={resume_batch}")

        # Run processing
        results = processor.process_reviews(
            review_data,
            callback=save_batch_callback,
            resume_from=resume_batch,
            checkpoint_name=checkpoint_name
        )
        
        print()
        print(f"✅ Analysis complete!")
        print()
        
        # Step 7: Update final status
        print("📊 Step 7: Finalizing...")
        print("-" * 70)
        
        with db_manager.get_session() as session:
            update_processing_status(
                session,
                status_id,
                len(reviews),
                status='completed'
            )
        
        print(f"✅ Processing status updated")
        print()
        
        # Final statistics
        print("="*70)
        print("🎉 Analysis Complete!")
        print("="*70)
        
        stats = processor.get_stats()
        print(f"\nReviews analyzed: {stats['total_processed']:,}")
        print(f"Aspects found: {stats['total_aspects_found']:,}")
        print(f"Errors: {stats['errors']}")
        
        if 'duration_seconds' in stats:
            print(f"Duration: {stats['duration_seconds']:.2f} seconds")
            print(f"Speed: {stats['reviews_per_second']:.2f} reviews/sec")
        
        print("\n" + "="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print("\nNext steps:")
        print(f"  1. Generate summaries: python scripts/generate_summaries.py --category {args.category}")
        print(f"  2. Launch UI: streamlit run src/ui/app.py")
        print()
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        
        # Update status to failed
        try:
            with db_manager.get_session() as session:
                update_processing_status(
                    session,
                    status_id,
                    processor.stats['total_processed'],
                    status='failed',
                    error='Interrupted by user'
                )
        except:
            pass
        
        return 1
    
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        logger.exception("Error during analysis")
        
        # Update status to failed
        try:
            with db_manager.get_session() as session:
                update_processing_status(
                    session,
                    status_id,
                    0,
                    status='failed',
                    error=str(e)
                )
        except:
            pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())

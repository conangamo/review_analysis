"""Test AI model with a single review."""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import AspectManager, ConfigLoader
from src.ai_engine import SentimentAnalyzer
from src.ai_engine.models import ZeroShotClassifier

print("\n" + "="*70)
print("🧪 AI Model Test")
print("="*70)

# Test review
test_review = """
This phone is amazing! The battery lasts all day even with heavy use.
The screen is bright and crisp, perfect for watching videos. 
However, the price is a bit steep for what you get. Overall, great value!
"""

print(f"\nTest Review:")
print(f"{test_review.strip()}")
print("\n" + "-"*70)

try:
    # Step 1: Load classifier
    print("\n1️⃣ Loading AI model...")
    start = time.time()
    
    classifier = ZeroShotClassifier(
        model_name="valhalla/distilbart-mnli-12-3",
        device="cuda",
        use_fp16=True,
        batch_size=32
    )
    
    load_time = time.time() - start
    print(f"✅ Model loaded in {load_time:.2f} seconds")
    
    # Step 2: Test simple classification
    print("\n2️⃣ Testing simple classification...")
    start = time.time()
    
    result = classifier.classify(
        "This is a positive review about battery life",
        ["positive", "negative", "neutral"]
    )
    
    classify_time = time.time() - start
    print(f"✅ Classification done in {classify_time:.2f} seconds")
    print(f"   Result: {result['labels'][0]} ({result['scores'][0]:.2%})")
    
    # Step 3: Initialize aspect manager
    print("\n3️⃣ Initializing aspect manager...")
    
    config_loader = ConfigLoader()
    aspect_manager = AspectManager("electronics", config_loader)
    
    print(f"✅ Loaded {len(aspect_manager.aspects)} aspects")
    
    # Step 4: Initialize sentiment analyzer
    print("\n4️⃣ Initializing sentiment analyzer...")
    
    analyzer = SentimentAnalyzer(
        "electronics",
        aspect_manager,
        classifier,
        use_keyword_filter=True
    )
    
    print(f"✅ Sentiment analyzer ready")
    
    # Step 5: Analyze test review
    print("\n5️⃣ Analyzing test review...")
    start = time.time()
    
    results = analyzer.analyze_review(test_review)
    
    analyze_time = time.time() - start
    print(f"✅ Analysis done in {analyze_time:.2f} seconds")
    
    # Display results
    print("\n" + "="*70)
    print("🎯 Analysis Results")
    print("="*70)
    
    if results:
        for result in results:
            print(f"\nAspect: {result['aspect']}")
            print(f"  Sentiment: {result['sentiment']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Tier: {result['tier']}")
    else:
        print("No aspects detected!")
    
    print("\n" + "="*70)
    print("✅ AI Model Test Complete!")
    print("="*70)
    
    # Performance summary
    print(f"\nPerformance:")
    print(f"  Model load: {load_time:.2f}s")
    print(f"  Single classification: {classify_time:.2f}s")
    print(f"  Full analysis: {analyze_time:.2f}s")
    print(f"  Aspects found: {len(results)}")
    
    # Estimate for 1000 reviews
    estimated_time = analyze_time * 1000
    print(f"\nEstimated for 1000 reviews: {estimated_time:.2f}s ({estimated_time/60:.1f} minutes)")
    
    print("\n✅ If you see this, AI model is working correctly!")
    print("   You can now run: python scripts/run_analysis.py --category electronics --limit 1000")
    print()

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n💡 Troubleshooting:")
    print("  1. Check GPU: python -c \"import torch; print(torch.cuda.is_available())\"")
    print("  2. Install PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("  3. Set USE_GPU=false in .env if no GPU")
    print()

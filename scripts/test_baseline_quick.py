"""
Quick test script để kiểm tra Baseline Model pipeline.

Chỉ train trên 1000 reviews để test nhanh.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import từ train_baseline
from train_baseline import (
    connect_db,
    load_data_from_db,
    create_baseline_model,
    DB_PATH
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def quick_test():
    """Quick test với sample nhỏ."""
    print("="*70)
    print("QUICK TEST - BASELINE MODEL")
    print("="*70)
    
    # Connect DB
    print("\n1. Connecting to database...")
    conn = connect_db(DB_PATH)
    print(f"   ✅ Connected: {DB_PATH}")
    
    # Load data
    print("\n2. Loading data...")
    df = load_data_from_db(conn)
    print(f"   ✅ Loaded {len(df):,} reviews")
    
    # Sample 1000 reviews
    if len(df) > 1000:
        df_sample = df.sample(n=1000, random_state=42)
        print(f"   📊 Sampled 1000 reviews for quick test")
    else:
        df_sample = df
    
    # Split
    print("\n3. Splitting train/test...")
    train_df, test_df = train_test_split(
        df_sample,
        test_size=0.2,
        random_state=42,
        stratify=df_sample['sentiment_label']
    )
    print(f"   ✅ Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Create and train model
    print("\n4. Training model...")
    model = create_baseline_model()
    
    X_train = train_df['text']
    y_train = train_df['sentiment_label']
    
    model.fit(X_train, y_train)
    print(f"   ✅ Training complete!")
    
    # Evaluate
    print("\n5. Evaluating...")
    X_test = test_df['text']
    y_test = test_df['sentiment_label']
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"   ✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n6. Classification Report:")
    print(classification_report(y_test, predictions, digits=4))
    
    # Test prediction
    print("\n7. Test predictions:")
    test_texts = [
        "This product is amazing! I love it!",
        "Terrible quality, waste of money.",
        "It's okay, nothing special."
    ]
    
    for text in test_texts:
        pred = model.predict([text])[0]
        proba = model.predict_proba([text])[0]
        conf = max(proba)
        print(f"   • Text: {text}")
        print(f"     → Prediction: {pred} (confidence: {conf:.4f})")
    
    conn.close()
    
    print("\n" + "="*70)
    print("✅ QUICK TEST COMPLETE!")
    print("="*70)
    print("\n💡 To run full training:")
    print("   python scripts/train_baseline.py")
    print("\n💡 To visualize and compare:")
    print("   python scripts/visualize_evaluation.py --compare")

if __name__ == "__main__":
    quick_test()

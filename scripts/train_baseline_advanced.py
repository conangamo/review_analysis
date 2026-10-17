"""
Script train Baseline Model với Custom Optimizer và Loss Function.

So sánh các cấu hình:
1. Default (LogReg + LBFGS + Log Loss)
2. SGD Optimizer với Learning Rate Scheduling
3. Focal Loss (via class weights) để xử lý class imbalance

Author: ABSA System
Date: 2026
"""

import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CẤU HÌNH
# ============================================================================

DB_PATH = "data/processed/reviews.db"
MODEL_DIR = "data/processed/baseline_models"
RESULTS_DIR = "data/processed/evaluation_results"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

def get_model_configs():
    """Định nghĩa các cấu hình model khác nhau."""
    configs = {
        # Config 1: Default (Baseline hiện tại)
        'default_lbfgs': {
            'name': 'Baseline_Default',
            'description': 'LogisticRegression + LBFGS + CrossEntropy',
            'model': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs',  # Default optimizer
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        },
        
        # Config 2: SGD Optimizer với Adaptive Learning Rate
        'sgd_adaptive': {
            'name': 'Baseline_SGD_Adaptive',
            'description': 'SGDClassifier + Adaptive LR + Log Loss',
            'model': SGDClassifier(
                loss='log_loss',              # Log loss = CrossEntropy
                penalty='l2',
                alpha=0.0001,
                learning_rate='adaptive',     # Adaptive LR
                eta0=0.01,                    # Initial LR
                max_iter=1000,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight='balanced'
            )
        },
        
        # Config 3: SGD với Optimal Learning Rate
        'sgd_optimal': {
            'name': 'Baseline_SGD_Optimal',
            'description': 'SGDClassifier + Optimal LR (Heuristic)',
            'model': SGDClassifier(
                loss='log_loss',
                penalty='elasticnet',          # L1 + L2
                l1_ratio=0.15,                 # Mix ratio
                alpha=0.0001,
                learning_rate='optimal',       # Optimal heuristic
                max_iter=1000,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight='balanced'
            )
        },
        
        # Config 4: Custom Class Weights (Focal Loss approximation)
        'custom_weights': {
            'name': 'Baseline_FocalLoss_Approx',
            'description': 'LogReg + Custom Weights (Focal Loss)',
            'model': 'custom',  # Will be created dynamically
            'compute_weights': True
        },
        
        # Config 5: SAGA Optimizer (Support L1)
        'saga_l1': {
            'name': 'Baseline_SAGA_L1',
            'description': 'LogReg + SAGA Optimizer + L1 Penalty',
            'model': LogisticRegression(
                max_iter=1000,
                penalty='l1',                  # L1 regularization
                solver='saga',                 # SAGA optimizer
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        }
    }
    
    return configs


def compute_focal_loss_weights(y_train, gamma=2.0):
    """
    Tính class weights xấp xỉ Focal Loss.
    
    Focal Loss: FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    
    Approximation: Tăng weight cho minority class mạnh hơn 'balanced'
    """
    class_counts = np.bincount(y_train)
    total = len(y_train)
    
    # Standard balanced weights
    balanced_weights = total / (len(class_counts) * class_counts)
    
    # Focal adjustment: (total/count)^gamma
    focal_weights = (total / class_counts) ** gamma
    
    # Normalize
    focal_weights = focal_weights / focal_weights.sum() * len(class_counts)
    
    return dict(enumerate(focal_weights))


def create_model_with_custom_weights(y_train):
    """Tạo model với custom class weights (Focal Loss approx)."""
    # Map string labels to numeric
    label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    y_numeric = np.array([label_map[label] for label in y_train])
    
    # Compute focal weights
    focal_weights = compute_focal_loss_weights(y_numeric, gamma=2.0)
    
    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        class_weight=focal_weights,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    return model


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_and_evaluate_config(config_name, config, train_df, test_df):
    """Train và evaluate một config."""
    print(f"\n{'='*70}")
    print(f"CONFIG: {config['name']}")
    print(f"{'='*70}")
    print(f"Description: {config['description']}")
    
    # Create pipeline
    if config['model'] == 'custom':
        # Custom weights config
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=2,
            max_df=0.95
        )
        
        X_train = train_df['text']
        y_train = train_df['sentiment_label']
        
        # Fit vectorizer
        X_train_vec = vectorizer.fit_transform(X_train)
        
        # Create model with custom weights
        classifier = create_model_with_custom_weights(y_train)
        
        # Train
        print(f"\n🚀 Training...")
        start_time = pd.Timestamp.now()
        classifier.fit(X_train_vec, y_train)
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Create pipeline for inference
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', classifier)
        ])
        
    else:
        # Standard pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=TFIDF_NGRAM_RANGE,
                min_df=2,
                max_df=0.95
            )),
            ('classifier', config['model'])
        ])
        
        X_train = train_df['text']
        y_train = train_df['sentiment_label']
        
        print(f"\n🚀 Training...")
        start_time = pd.Timestamp.now()
        pipeline.fit(X_train, y_train)
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    
    print(f"   ✅ Training completed in {elapsed:.2f}s")
    
    # Evaluate
    X_test = test_df['text']
    y_test = test_df['sentiment_label']
    
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\n📊 Classification Report:")
    print(classification_report(
        y_test, 
        y_pred,
        labels=['positive', 'neutral', 'negative'],
        target_names=['Positive', 'Neutral', 'Negative'],
        digits=4
    ))
    
    return {
        'config_name': config_name,
        'model_name': config['name'],
        'description': config['description'],
        'accuracy': accuracy,
        'training_time': elapsed,
        'pipeline': pipeline,
        'y_test': y_test,
        'y_pred': y_pred
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""
    print("="*70)
    print("BASELINE MODEL - CUSTOM OPTIMIZER & LOSS COMPARISON")
    print("="*70)
    
    # Load data
    print(f"\n📊 Loading data from {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT id, text, rating
    FROM reviews
    WHERE text IS NOT NULL AND rating IS NOT NULL
        AND LENGTH(text) >= 20
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    def rating_to_sentiment(rating):
        if rating <= 2.0:
            return 'negative'
        elif rating >= 4.0:
            return 'positive'
        else:
            return 'neutral'
    
    df['sentiment_label'] = df['rating'].apply(rating_to_sentiment)
    
    print(f"   ✅ Loaded {len(df):,} reviews")
    
    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['sentiment_label']
    )
    
    print(f"   • Train: {len(train_df):,}")
    print(f"   • Test:  {len(test_df):,}")
    
    # Get configurations
    configs = get_model_configs()
    
    # Train all configs
    results = []
    
    for config_name, config in configs.items():
        result = train_and_evaluate_config(config_name, config, train_df, test_df)
        results.append(result)
    
    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Description': r['description'],
            'Accuracy': f"{r['accuracy']:.4f}",
            'Training Time (s)': f"{r['training_time']:.2f}"
        }
        for r in results
    ])
    
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print(f"\n🏆 BEST MODEL: {best_result['model_name']}")
    print(f"   • Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"   • Training Time: {best_result['training_time']:.2f}s")
    
    # Export results
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_file = output_dir / f'optimizer_loss_comparison_{timestamp}.csv'
    comparison_df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
    
    print(f"\n📄 Exported comparison: {comparison_file}")
    
    # Save best model
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_file = model_dir / f"{best_result['model_name']}_{timestamp}.pkl"
    joblib.dump(best_result['pipeline'], best_model_file)
    
    print(f"💾 Saved best model: {best_model_file}")
    
    print(f"\n{'='*70}")
    print("✅ COMPLETED!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

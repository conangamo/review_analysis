"""
Script tạo Confusion Matrix chi tiết cho ABSA Model.

Không cần training, chỉ analyze predictions đã có.

Author: ABSA System
Date: 2026
"""

import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CẤU HÌNH
# ============================================================================

DB_PATH = "data/processed/reviews.db"
OUTPUT_DIR = "data/processed/evaluation_results"


# ============================================================================
# FUNCTIONS
# ============================================================================

def rating_to_sentiment(rating: float) -> str:
    """Convert rating to ground truth sentiment."""
    if rating <= 2.0:
        return 'negative'
    elif rating >= 4.0:
        return 'positive'
    else:
        return 'neutral'


def load_absa_predictions(db_path: str) -> pd.DataFrame:
    """Load ABSA model predictions from database."""
    conn = sqlite3.connect(db_path)
    
    # Aggregate: lấy sentiment dominant cho mỗi review
    query = """
    SELECT 
        a.review_id,
        r.rating,
        a.sentiment,
        SUM(a.confidence_score) as total_confidence
    FROM aspect_sentiments a
    INNER JOIN reviews r ON a.review_id = r.id
    WHERE r.rating IS NOT NULL
    GROUP BY a.review_id, a.sentiment
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Get dominant sentiment for each review
    idx = df.groupby('review_id')['total_confidence'].idxmax()
    df_dominant = df.loc[idx].reset_index(drop=True)
    
    # Add ground truth
    df_dominant['ground_truth'] = df_dominant['rating'].apply(rating_to_sentiment)
    df_dominant['predicted'] = df_dominant['sentiment']
    
    return df_dominant


def plot_confusion_matrix(y_true, y_pred, output_path: Path):
    """Vẽ confusion matrix với seaborn."""
    # Compute confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=['positive', 'neutral', 'negative']
    )
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Absolute counts
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Positive', 'Neutral', 'Negative'],
        yticklabels=['Positive', 'Neutral', 'Negative'],
        ax=ax1,
        cbar_kws={'label': 'Count'}
    )
    ax1.set_ylabel('Ground Truth (from Rating)', fontsize=12)
    ax1.set_xlabel('ABSA Model Prediction', fontsize=12)
    ax1.set_title('Confusion Matrix - Absolute Counts', fontsize=14, fontweight='bold')
    
    # Plot 2: Normalized (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        xticklabels=['Positive', 'Neutral', 'Negative'],
        yticklabels=['Positive', 'Neutral', 'Negative'],
        ax=ax2,
        cbar_kws={'label': 'Percentage (%)'},
        vmin=0,
        vmax=100
    )
    ax2.set_ylabel('Ground Truth (from Rating)', fontsize=12)
    ax2.set_xlabel('ABSA Model Prediction', fontsize=12)
    ax2.set_title('Confusion Matrix - Normalized (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   📊 Saved: {output_path}")
    
    return cm, cm_normalized


def analyze_confusion_matrix(cm, labels=['positive', 'neutral', 'negative']):
    """Phân tích chi tiết confusion matrix."""
    print(f"\n{'='*70}")
    print("CONFUSION MATRIX ANALYSIS")
    print(f"{'='*70}")
    
    total = cm.sum()
    
    print(f"\n📊 Overall Stats:")
    print(f"   • Total predictions: {total:,}")
    print(f"   • Correct: {np.trace(cm):,} ({np.trace(cm)/total*100:.2f}%)")
    print(f"   • Incorrect: {total - np.trace(cm):,} ({(total - np.trace(cm))/total*100:.2f}%)")
    
    print(f"\n📊 Per-Class Analysis:")
    for i, label in enumerate(labels):
        true_positives = cm[i, i]
        false_positives = cm[:, i].sum() - cm[i, i]
        false_negatives = cm[i, :].sum() - cm[i, i]
        
        total_actual = cm[i, :].sum()
        total_predicted = cm[:, i].sum()
        
        precision = true_positives / total_predicted if total_predicted > 0 else 0
        recall = true_positives / total_actual if total_actual > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n   {label.upper()}:")
        print(f"      • Actual count: {total_actual:,}")
        print(f"      • Predicted count: {total_predicted:,}")
        print(f"      • True Positives: {true_positives:,}")
        print(f"      • False Positives: {false_positives:,}")
        print(f"      • False Negatives: {false_negatives:,}")
        print(f"      • Precision: {precision:.4f}")
        print(f"      • Recall: {recall:.4f}")
        print(f"      • F1-Score: {f1:.4f}")
    
    # Common misclassifications
    print(f"\n📊 Common Misclassifications:")
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    
    top_errors = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                top_errors.append({
                    'actual': labels[i],
                    'predicted': labels[j],
                    'count': cm[i, j],
                    'percentage': cm[i, j] / cm[i, :].sum() * 100
                })
    
    top_errors = sorted(top_errors, key=lambda x: x['count'], reverse=True)[:5]
    
    for idx, error in enumerate(top_errors, 1):
        print(f"   {idx}. Actual {error['actual'].upper()} → "
              f"Predicted {error['predicted'].upper()}: "
              f"{error['count']:,} cases ({error['percentage']:.1f}%)")


def export_confusion_matrix_csv(cm, cm_normalized, output_dir: Path):
    """Export confusion matrix to CSV."""
    labels = ['Positive', 'Neutral', 'Negative']
    
    # Absolute counts
    df_abs = pd.DataFrame(
        cm,
        index=[f'Actual_{label}' for label in labels],
        columns=[f'Predicted_{label}' for label in labels]
    )
    
    # Normalized percentages
    df_norm = pd.DataFrame(
        cm_normalized,
        index=[f'Actual_{label}' for label in labels],
        columns=[f'Predicted_{label}' for label in labels]
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Export
    abs_file = output_dir / f'absa_confusion_matrix_counts_{timestamp}.csv'
    norm_file = output_dir / f'absa_confusion_matrix_normalized_{timestamp}.csv'
    
    df_abs.to_csv(abs_file, encoding='utf-8-sig')
    df_norm.to_csv(norm_file, encoding='utf-8-sig')
    
    print(f"\n   📄 Saved: {abs_file}")
    print(f"   📄 Saved: {norm_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""
    print("="*70)
    print("CONFUSION MATRIX - ABSA MODEL")
    print("="*70)
    
    # Load predictions
    print(f"\n📊 Loading ABSA predictions from database...")
    df = load_absa_predictions(DB_PATH)
    
    print(f"   ✅ Loaded {len(df):,} reviews with predictions")
    
    # Distribution
    print(f"\n📊 Ground Truth Distribution:")
    gt_counts = df['ground_truth'].value_counts()
    for sentiment, count in gt_counts.items():
        print(f"   • {sentiment.capitalize():8s}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print(f"\n📊 Prediction Distribution:")
    pred_counts = df['predicted'].value_counts()
    for sentiment, count in pred_counts.items():
        print(f"   • {sentiment.capitalize():8s}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot confusion matrix
    print(f"\n📊 Creating confusion matrix visualization...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'absa_confusion_matrix_{timestamp}.png'
    
    cm, cm_normalized = plot_confusion_matrix(
        df['ground_truth'],
        df['predicted'],
        plot_path
    )
    
    # Analyze
    analyze_confusion_matrix(cm)
    
    # Export CSV
    print(f"\n📄 Exporting confusion matrix to CSV...")
    export_confusion_matrix_csv(cm, cm_normalized, output_dir)
    
    print(f"\n{'='*70}")
    print("✅ COMPLETED!")
    print(f"{'='*70}")
    
    print(f"\n💡 Files created:")
    print(f"   • Visualization: {plot_path}")
    print(f"   • CSV files in: {output_dir}")
    
    print(f"\n💡 Use in report:")
    print(f"   1. Import confusion matrix image")
    print(f"   2. Discuss misclassification patterns")
    print(f"   3. Analyze which classes are confused")


if __name__ == "__main__":
    main()

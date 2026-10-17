"""
Script đánh giá chi tiết hiệu suất của cả ABSA Model và Baseline Model.

Tính toán các metrics:
- Precision, Recall, F1-Score cho TỪNG nhãn (Positive, Negative, Neutral)
- Macro F1-Score
- Weighted F1-Score
- Overall Accuracy

Xuất kết quả ra file CSV để đưa vào báo cáo.

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

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# CẤU HÌNH
# ============================================================================

DB_PATH = "data/processed/reviews.db"
OUTPUT_DIR = "data/processed/evaluation_results"

# Định nghĩa thứ tự các sentiment classes
SENTIMENT_CLASSES = ['positive', 'neutral', 'negative']


# ============================================================================
# HÀM KẾT NỐI DATABASE
# ============================================================================

def connect_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Kết nối đến SQLite database."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database không tồn tại: {db_path}")
    return sqlite3.connect(db_path)


# ============================================================================
# HÀM LOAD DỮ LIỆU VÀ GROUND TRUTH
# ============================================================================

def rating_to_sentiment(rating: float) -> str:
    """
    Chuyển rating (1-5) thành sentiment label (ground truth).
    
    Args:
        rating: Rating từ 1.0 đến 5.0
        
    Returns:
        Sentiment label: 'positive', 'negative', 'neutral'
    """
    if rating <= 2.0:
        return 'negative'
    elif rating >= 4.0:
        return 'positive'
    else:
        return 'neutral'


def load_baseline_predictions(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load dự đoán của Baseline Model và ground truth.
    
    Args:
        conn: Database connection
        
    Returns:
        DataFrame với cột: review_id, rating, ground_truth, predicted
    """
    query = """
    SELECT 
        bp.review_id,
        r.rating,
        bp.predicted_sentiment,
        bp.confidence_score
    FROM baseline_predictions bp
    INNER JOIN reviews r ON bp.review_id = r.id
    WHERE bp.model_name = 'Baseline_ML_Model'
        AND r.rating IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    
    if len(df) == 0:
        return None
    
    # Tạo ground truth từ rating
    df['ground_truth'] = df['rating'].apply(rating_to_sentiment)
    df['predicted'] = df['predicted_sentiment']
    
    return df[['review_id', 'rating', 'ground_truth', 'predicted', 'confidence_score']]


def load_absa_predictions(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load dự đoán của ABSA Model và ground truth.
    
    ABSA model có nhiều aspects cho mỗi review, lấy sentiment phổ biến nhất
    hoặc trung bình weighted by confidence.
    
    Args:
        conn: Database connection
        
    Returns:
        DataFrame với cột: review_id, rating, ground_truth, predicted
    """
    query = """
    SELECT 
        a.review_id,
        r.rating,
        a.sentiment,
        a.confidence_score
    FROM aspect_sentiments a
    INNER JOIN reviews r ON a.review_id = r.id
    WHERE r.rating IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    
    if len(df) == 0:
        return None
    
    # Aggregate: lấy sentiment có tổng confidence cao nhất cho mỗi review
    def get_dominant_sentiment(group):
        sentiment_scores = group.groupby('sentiment')['confidence_score'].sum()
        return sentiment_scores.idxmax()
    
    def get_avg_confidence(group):
        return group['confidence_score'].mean()
    
    # Group by review_id và aggregate
    aggregated = df.groupby('review_id').apply(
        lambda x: pd.Series({
            'rating': x['rating'].iloc[0],
            'predicted': get_dominant_sentiment(x),
            'confidence_score': get_avg_confidence(x)
        })
    ).reset_index()
    
    # Tạo ground truth từ rating
    aggregated['ground_truth'] = aggregated['rating'].apply(rating_to_sentiment)
    
    return aggregated[['review_id', 'rating', 'ground_truth', 'predicted', 'confidence_score']]


# ============================================================================
# HÀM TÍNH METRICS CHI TIẾT
# ============================================================================

def calculate_detailed_metrics(
    y_true: List[str],
    y_pred: List[str],
    model_name: str
) -> pd.DataFrame:
    """
    Tính toán Precision, Recall, F1-Score chi tiết cho từng class.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_name: Tên model
        
    Returns:
        DataFrame với metrics chi tiết
    """
    # Tính precision, recall, f1-score cho từng class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=SENTIMENT_CLASSES,
        zero_division=0
    )
    
    # Tính macro average
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Tính weighted average
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='weighted',
        zero_division=0
    )
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Tạo DataFrame cho từng class
    metrics_data = []
    
    for i, sentiment in enumerate(SENTIMENT_CLASSES):
        metrics_data.append({
            'Model': model_name,
            'Sentiment_Class': sentiment.capitalize(),
            'Precision': precision[i],
            'Recall': recall[i],
            'F1_Score': f1[i],
            'Support': support[i]
        })
    
    # Thêm Macro average
    metrics_data.append({
        'Model': model_name,
        'Sentiment_Class': 'Macro Avg',
        'Precision': macro_precision,
        'Recall': macro_recall,
        'F1_Score': macro_f1,
        'Support': sum(support)
    })
    
    # Thêm Weighted average
    metrics_data.append({
        'Model': model_name,
        'Sentiment_Class': 'Weighted Avg',
        'Precision': weighted_precision,
        'Recall': weighted_recall,
        'F1_Score': weighted_f1,
        'Support': sum(support)
    })
    
    # Thêm Overall accuracy
    metrics_data.append({
        'Model': model_name,
        'Sentiment_Class': 'Accuracy',
        'Precision': accuracy,
        'Recall': accuracy,
        'F1_Score': accuracy,
        'Support': sum(support)
    })
    
    return pd.DataFrame(metrics_data)


def print_classification_report(
    y_true: List[str],
    y_pred: List[str],
    model_name: str
):
    """
    In classification report đầy đủ.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_name: Tên model
    """
    print(f"\n{'='*70}")
    print(f"CLASSIFICATION REPORT - {model_name}")
    print(f"{'='*70}")
    
    print(classification_report(
        y_true,
        y_pred,
        labels=SENTIMENT_CLASSES,
        target_names=[s.capitalize() for s in SENTIMENT_CLASSES],
        digits=4,
        zero_division=0
    ))
    
    # Confusion Matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=SENTIMENT_CLASSES)
    
    print(f"\n                Predicted")
    print(f"                {'  '.join([s[:3].capitalize() for s in SENTIMENT_CLASSES])}")
    for i, sentiment in enumerate(SENTIMENT_CLASSES):
        row_values = '  '.join([f'{cm[i,j]:5d}' for j in range(len(SENTIMENT_CLASSES))])
        print(f"   Actual {sentiment[:3].capitalize()}  {row_values}")


# ============================================================================
# HÀM XUẤT KẾT QUẢ
# ============================================================================

def export_detailed_metrics(
    baseline_metrics: pd.DataFrame,
    absa_metrics: pd.DataFrame,
    output_dir: str = OUTPUT_DIR
) -> str:
    """
    Xuất metrics chi tiết ra file CSV.
    
    Args:
        baseline_metrics: DataFrame metrics của Baseline Model
        absa_metrics: DataFrame metrics của ABSA Model
        output_dir: Thư mục output
        
    Returns:
        Đường dẫn file CSV
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Kết hợp 2 DataFrames
    combined_metrics = pd.concat([baseline_metrics, absa_metrics], ignore_index=True)
    
    # Export CSV
    csv_filename = f'detailed_metrics_comparison_{timestamp}.csv'
    csv_path = output_path / csv_filename
    
    combined_metrics.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    return str(csv_path)


def export_comparison_summary(
    baseline_metrics: pd.DataFrame,
    absa_metrics: pd.DataFrame,
    output_dir: str = OUTPUT_DIR
) -> str:
    """
    Xuất bảng so sánh tóm tắt (side-by-side).
    
    Args:
        baseline_metrics: DataFrame metrics của Baseline Model
        absa_metrics: DataFrame metrics của ABSA Model
        output_dir: Thư mục output
        
    Returns:
        Đường dẫn file CSV
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Tạo pivot table để so sánh side-by-side
    summary_data = {
        'Metric': [],
        'Baseline_Precision': [],
        'Baseline_Recall': [],
        'Baseline_F1': [],
        'ABSA_Precision': [],
        'ABSA_Recall': [],
        'ABSA_F1': []
    }
    
    for sentiment_class in SENTIMENT_CLASSES + ['Macro Avg', 'Weighted Avg', 'Accuracy']:
        baseline_row = baseline_metrics[
            baseline_metrics['Sentiment_Class'] == sentiment_class
        ].iloc[0] if sentiment_class in baseline_metrics['Sentiment_Class'].values else None
        
        absa_row = absa_metrics[
            absa_metrics['Sentiment_Class'] == sentiment_class
        ].iloc[0] if sentiment_class in absa_metrics['Sentiment_Class'].values else None
        
        if baseline_row is not None and absa_row is not None:
            summary_data['Metric'].append(sentiment_class)
            summary_data['Baseline_Precision'].append(f"{baseline_row['Precision']:.4f}")
            summary_data['Baseline_Recall'].append(f"{baseline_row['Recall']:.4f}")
            summary_data['Baseline_F1'].append(f"{baseline_row['F1_Score']:.4f}")
            summary_data['ABSA_Precision'].append(f"{absa_row['Precision']:.4f}")
            summary_data['ABSA_Recall'].append(f"{absa_row['Recall']:.4f}")
            summary_data['ABSA_F1'].append(f"{absa_row['F1_Score']:.4f}")
    
    summary_df = pd.DataFrame(summary_data)
    
    # Export CSV
    csv_filename = f'metrics_comparison_summary_{timestamp}.csv'
    csv_path = output_path / csv_filename
    
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    return str(csv_path)


# ============================================================================
# HÀM MAIN
# ============================================================================

def main():
    """Hàm main thực thi evaluation."""
    print("="*70)
    print("ĐÁNH GIÁ CHI TIẾT - BASELINE MODEL vs ABSA MODEL")
    print("="*70)
    print("\nMetrics: Precision, Recall, F1-Score")
    print("Classes: Positive, Neutral, Negative")
    print(f"Database: {DB_PATH}")
    
    try:
        # Kết nối database
        print(f"\n{'='*70}")
        print("BƯỚC 1: KẾT NỐI DATABASE VÀ LOAD DỮ LIỆU")
        print(f"{'='*70}")
        
        conn = connect_db(DB_PATH)
        print(f"✅ Đã kết nối: {DB_PATH}")
        
        # Load Baseline predictions
        print(f"\n📊 Đang load dự đoán của Baseline Model...")
        baseline_df = load_baseline_predictions(conn)
        
        if baseline_df is None or len(baseline_df) == 0:
            print(f"\n❌ Không tìm thấy dự đoán của Baseline Model!")
            print(f"   Chạy trước: python scripts/train_baseline.py")
            return 1
        
        print(f"   ✅ Đã load {len(baseline_df):,} reviews từ Baseline Model")
        
        # Load ABSA predictions
        print(f"\n📊 Đang load dự đoán của ABSA Model...")
        absa_df = load_absa_predictions(conn)
        
        if absa_df is None or len(absa_df) == 0:
            print(f"\n❌ Không tìm thấy dự đoán của ABSA Model!")
            print(f"   Chạy trước: python scripts/run_analysis.py --category <category>")
            return 1
        
        print(f"   ✅ Đã load {len(absa_df):,} reviews từ ABSA Model")
        
        # Tìm common reviews (reviews có trong cả 2 models)
        common_review_ids = set(baseline_df['review_id']) & set(absa_df['review_id'])
        
        if len(common_review_ids) == 0:
            print(f"\n⚠️  Không có reviews chung giữa 2 models!")
            print(f"   Baseline có {len(baseline_df):,} reviews")
            print(f"   ABSA có {len(absa_df):,} reviews")
            print(f"\n💡 Đề xuất: Chạy lại cả 2 models trên cùng dataset")
            
            # Tiếp tục đánh giá riêng từng model
            print(f"\n⚙️  Đánh giá riêng từng model...")
            
        else:
            print(f"\n✅ Tìm thấy {len(common_review_ids):,} reviews chung")
            
            # Filter chỉ lấy common reviews
            baseline_df = baseline_df[baseline_df['review_id'].isin(common_review_ids)]
            absa_df = absa_df[absa_df['review_id'].isin(common_review_ids)]
            
            # Sort theo review_id để đảm bảo thứ tự giống nhau
            baseline_df = baseline_df.sort_values('review_id').reset_index(drop=True)
            absa_df = absa_df.sort_values('review_id').reset_index(drop=True)
        
        conn.close()
        
        # Đánh giá Baseline Model
        print(f"\n{'='*70}")
        print("BƯỚC 2: ĐÁNH GIÁ BASELINE MODEL")
        print(f"{'='*70}")
        
        baseline_y_true = baseline_df['ground_truth'].tolist()
        baseline_y_pred = baseline_df['predicted'].tolist()
        
        print_classification_report(
            baseline_y_true,
            baseline_y_pred,
            'Baseline Model (TF-IDF + Logistic Regression)'
        )
        
        baseline_metrics = calculate_detailed_metrics(
            baseline_y_true,
            baseline_y_pred,
            'Baseline'
        )
        
        # Đánh giá ABSA Model
        print(f"\n{'='*70}")
        print("BƯỚC 3: ĐÁNH GIÁ ABSA MODEL")
        print(f"{'='*70}")
        
        absa_y_true = absa_df['ground_truth'].tolist()
        absa_y_pred = absa_df['predicted'].tolist()
        
        print_classification_report(
            absa_y_true,
            absa_y_pred,
            'ABSA Model (Zero-shot Classification)'
        )
        
        absa_metrics = calculate_detailed_metrics(
            absa_y_true,
            absa_y_pred,
            'ABSA'
        )
        
        # So sánh
        print(f"\n{'='*70}")
        print("BƯỚC 4: SO SÁNH TỔNG QUÁT")
        print(f"{'='*70}")
        
        # Lấy Macro F1
        baseline_macro_f1 = baseline_metrics[
            baseline_metrics['Sentiment_Class'] == 'Macro Avg'
        ]['F1_Score'].iloc[0]
        
        absa_macro_f1 = absa_metrics[
            absa_metrics['Sentiment_Class'] == 'Macro Avg'
        ]['F1_Score'].iloc[0]
        
        # Lấy Overall Accuracy
        baseline_acc = baseline_metrics[
            baseline_metrics['Sentiment_Class'] == 'Accuracy'
        ]['F1_Score'].iloc[0]
        
        absa_acc = absa_metrics[
            absa_metrics['Sentiment_Class'] == 'Accuracy'
        ]['F1_Score'].iloc[0]
        
        print(f"\n📊 Macro F1-Score:")
        print(f"   • Baseline Model: {baseline_macro_f1:.4f} ({baseline_macro_f1*100:.2f}%)")
        print(f"   • ABSA Model:     {absa_macro_f1:.4f} ({absa_macro_f1*100:.2f}%)")
        print(f"   • Chênh lệch:     {(absa_macro_f1 - baseline_macro_f1):+.4f} ({(absa_macro_f1 - baseline_macro_f1)*100:+.2f}%)")
        
        print(f"\n📊 Overall Accuracy:")
        print(f"   • Baseline Model: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
        print(f"   • ABSA Model:     {absa_acc:.4f} ({absa_acc*100:.2f}%)")
        print(f"   • Chênh lệch:     {(absa_acc - baseline_acc):+.4f} ({(absa_acc - baseline_acc)*100:+.2f}%)")
        
        # So sánh từng class
        print(f"\n📊 F1-Score theo từng Class:")
        for sentiment in SENTIMENT_CLASSES:
            baseline_f1 = baseline_metrics[
                baseline_metrics['Sentiment_Class'] == sentiment.capitalize()
            ]['F1_Score'].iloc[0]
            
            absa_f1 = absa_metrics[
                absa_metrics['Sentiment_Class'] == sentiment.capitalize()
            ]['F1_Score'].iloc[0]
            
            diff = absa_f1 - baseline_f1
            print(f"   • {sentiment.capitalize():8s}: Baseline={baseline_f1:.4f}, "
                  f"ABSA={absa_f1:.4f}, Diff={diff:+.4f}")
        
        # Xuất kết quả
        print(f"\n{'='*70}")
        print("BƯỚC 5: XUẤT KẾT QUẢ")
        print(f"{'='*70}")
        
        # Export detailed metrics
        detailed_csv = export_detailed_metrics(baseline_metrics, absa_metrics)
        print(f"\n✅ Đã xuất file detailed metrics:")
        print(f"   📄 {detailed_csv}")
        
        # Export comparison summary
        summary_csv = export_comparison_summary(baseline_metrics, absa_metrics)
        print(f"\n✅ Đã xuất file comparison summary:")
        print(f"   📄 {summary_csv}")
        
        # Preview detailed metrics
        print(f"\n📋 Preview - Detailed Metrics Comparison:")
        print(f"\n{pd.concat([baseline_metrics, absa_metrics], ignore_index=True).to_string(index=False)}")
        
        # Kết luận
        print(f"\n{'='*70}")
        print("✅ HOÀN THÀNH ĐÁNH GIÁ!")
        print(f"{'='*70}")
        
        print(f"\n📊 Tổng kết:")
        print(f"   • Dataset size: {len(baseline_df):,} reviews")
        print(f"   • Classes: {', '.join([s.capitalize() for s in SENTIMENT_CLASSES])}")
        print(f"   • Baseline Macro F1: {baseline_macro_f1:.4f}")
        print(f"   • ABSA Macro F1:     {absa_macro_f1:.4f}")
        
        if absa_macro_f1 > baseline_macro_f1:
            improvement = (absa_macro_f1 - baseline_macro_f1) * 100
            print(f"\n🏆 ABSA Model vượt trội hơn {improvement:.2f}% Macro F1!")
        else:
            print(f"\n⚠️  Baseline Model có performance tương đương hoặc tốt hơn")
        
        print(f"\n💡 File đã được xuất, có thể đưa vào báo cáo:")
        print(f"   • {detailed_csv}")
        print(f"   • {summary_csv}")
        
        print(f"\n{'='*70}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

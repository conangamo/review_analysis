"""
Script train và đánh giá Baseline Model sử dụng TF-IDF + Logistic Regression.

Mục tiêu: Tạo baseline để so sánh với mô hình ABSA hiện tại.

Pipeline:
1. Đọc dữ liệu từ database (reviews.db)
2. Train model TF-IDF + Logistic Regression
3. Dự đoán sentiment trên tập test
4. Lưu kết quả với định danh "Baseline_ML_Model"
5. Tính toán metrics để so sánh

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

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# CẤU HÌNH
# ============================================================================

DB_PATH = "data/processed/reviews.db"
MODEL_DIR = "data/processed/baseline_models"
RESULTS_DIR = "data/processed/evaluation_results"

# Model identifier
MODEL_NAME = "Baseline_ML_Model"

# Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)  # unigrams + bigrams


# ============================================================================
# HÀM KẾT NỐI DATABASE
# ============================================================================

def connect_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Kết nối đến SQLite database."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database không tồn tại: {db_path}")
    return sqlite3.connect(db_path)


# ============================================================================
# HÀM LOAD DỮ LIỆU
# ============================================================================

def load_data_from_db(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load reviews và rating từ database.
    
    Args:
        conn: Database connection
        
    Returns:
        DataFrame với cột: id, text, rating, sentiment_label
    """
    print("\n📊 Đang load dữ liệu từ database...")
    
    query = """
    SELECT 
        id,
        text,
        rating
    FROM reviews
    WHERE text IS NOT NULL 
        AND rating IS NOT NULL
        AND LENGTH(text) >= 20
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Convert rating thành sentiment label (ground truth dựa trên rating)
    def rating_to_sentiment(rating: float) -> str:
        """Convert rating (1-5) sang sentiment label."""
        if rating <= 2.0:
            return 'negative'
        elif rating >= 4.0:
            return 'positive'
        else:
            return 'neutral'
    
    df['sentiment_label'] = df['rating'].apply(rating_to_sentiment)
    
    print(f"   ✅ Đã load {len(df):,} reviews")
    print(f"\n   Phân bố sentiment (dựa trên rating):")
    sentiment_counts = df['sentiment_label'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df) * 100)
        print(f"      • {sentiment.capitalize():8s}: {count:6,d} ({percentage:5.2f}%)")
    
    return df


# ============================================================================
# HÀM TRAIN MODEL
# ============================================================================

def create_baseline_model() -> Pipeline:
    """
    Tạo baseline model pipeline: TF-IDF + Logistic Regression.
    
    Returns:
        sklearn Pipeline object
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=2,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    
    return pipeline


def train_baseline_model(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """
    Train baseline model và split train/test.
    
    Args:
        df: DataFrame với text và sentiment_label
        test_size: Tỷ lệ test set
        
    Returns:
        Tuple (trained_model, train_df, test_df)
    """
    print("\n" + "="*70)
    print("TRAINING BASELINE MODEL")
    print("="*70)
    
    # Split train/test
    print(f"\n📊 Chia train/test với tỷ lệ {int((1-test_size)*100)}/{int(test_size*100)}...")
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df['sentiment_label']
    )
    
    print(f"   ✅ Train set: {len(train_df):,} reviews")
    print(f"   ✅ Test set:  {len(test_df):,} reviews")
    
    # Tạo model pipeline
    print(f"\n🤖 Khởi tạo model pipeline...")
    print(f"   • TF-IDF: max_features={TFIDF_MAX_FEATURES}, ngram={TFIDF_NGRAM_RANGE}")
    print(f"   • Classifier: Logistic Regression (balanced)")
    
    model = create_baseline_model()
    
    # Train model
    print(f"\n🚀 Bắt đầu training...")
    start_time = pd.Timestamp.now()
    
    X_train = train_df['text']
    y_train = train_df['sentiment_label']
    
    model.fit(X_train, y_train)
    
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    print(f"   ✅ Training hoàn thành trong {elapsed:.2f} giây!")
    
    # Đánh giá trên train set
    print(f"\n📈 Đánh giá trên train set...")
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"   • Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    # Đánh giá trên test set
    print(f"\n📈 Đánh giá trên test set...")
    X_test = test_df['text']
    y_test = test_df['sentiment_label']
    
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"   • Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Classification report
    print(f"\n📊 Classification Report (Test Set):")
    print(classification_report(
        y_test, 
        test_pred, 
        labels=['positive', 'neutral', 'negative'],
        target_names=['Positive', 'Neutral', 'Negative'],
        digits=4
    ))
    
    # Confusion matrix
    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, test_pred, labels=['positive', 'neutral', 'negative'])
    print(f"                Predicted")
    print(f"                Pos    Neu    Neg")
    print(f"   Actual Pos  {cm[0,0]:5d}  {cm[0,1]:5d}  {cm[0,2]:5d}")
    print(f"          Neu  {cm[1,0]:5d}  {cm[1,1]:5d}  {cm[1,2]:5d}")
    print(f"          Neg  {cm[2,0]:5d}  {cm[2,1]:5d}  {cm[2,2]:5d}")
    
    # Export detailed metrics to CSV
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        test_pred,
        labels=['positive', 'neutral', 'negative'],
        zero_division=0
    )
    
    metrics_data = []
    for i, sentiment in enumerate(['Positive', 'Neutral', 'Negative']):
        metrics_data.append({
            'Model': 'Baseline',
            'Sentiment_Class': sentiment,
            'Precision': f"{precision[i]:.4f}",
            'Recall': f"{recall[i]:.4f}",
            'F1_Score': f"{f1[i]:.4f}",
            'Support': support[i]
        })
    
    # Macro average
    metrics_data.append({
        'Model': 'Baseline',
        'Sentiment_Class': 'Macro Avg',
        'Precision': f"{np.mean(precision):.4f}",
        'Recall': f"{np.mean(recall):.4f}",
        'F1_Score': f"{np.mean(f1):.4f}",
        'Support': sum(support)
    })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Save metrics
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_file = output_dir / f'baseline_detailed_metrics_{timestamp}.csv'
    metrics_df.to_csv(metrics_file, index=False, encoding='utf-8-sig')
    
    print(f"\n📄 Đã xuất detailed metrics: {metrics_file}")
    
    return model, train_df, test_df


# ============================================================================
# HÀM DỰ ĐOÁN VÀ LƯU KẾT QUẢ
# ============================================================================

def predict_and_save_results(
    model: Pipeline,
    test_df: pd.DataFrame,
    conn: sqlite3.Connection,
    model_name: str = MODEL_NAME
) -> pd.DataFrame:
    """
    Dự đoán sentiment và lưu kết quả vào database.
    
    Args:
        model: Trained model pipeline
        test_df: Test DataFrame
        conn: Database connection
        model_name: Tên định danh của model
        
    Returns:
        DataFrame chứa kết quả dự đoán
    """
    print("\n" + "="*70)
    print("DỰ ĐOÁN VÀ LƯU KẾT QUẢ")
    print("="*70)
    
    print(f"\n🔮 Đang dự đoán trên {len(test_df):,} reviews...")
    
    # Predict sentiment
    X_test = test_df['text']
    predictions = model.predict(X_test)
    
    # Get probability scores (confidence)
    probabilities = model.predict_proba(X_test)
    confidence_scores = np.max(probabilities, axis=1)
    
    # Tạo DataFrame kết quả
    results_df = test_df.copy()
    results_df['predicted_sentiment'] = predictions
    results_df['confidence_score'] = confidence_scores
    results_df['model_name'] = model_name
    results_df['predicted_at'] = datetime.now()
    
    print(f"   ✅ Hoàn thành dự đoán!")
    print(f"\n   Phân bố sentiment dự đoán:")
    pred_counts = results_df['predicted_sentiment'].value_counts()
    for sentiment, count in pred_counts.items():
        percentage = (count / len(results_df) * 100)
        print(f"      • {sentiment.capitalize():8s}: {count:6,d} ({percentage:5.2f}%)")
    
    print(f"\n   Confidence score stats:")
    print(f"      • Mean: {results_df['confidence_score'].mean():.4f}")
    print(f"      • Median: {results_df['confidence_score'].median():.4f}")
    print(f"      • Min: {results_df['confidence_score'].min():.4f}")
    print(f"      • Max: {results_df['confidence_score'].max():.4f}")
    
    # Lưu vào database (bảng mới: baseline_predictions)
    print(f"\n💾 Đang lưu kết quả vào database...")
    
    # Tạo bảng nếu chưa tồn tại
    create_table_query = """
    CREATE TABLE IF NOT EXISTS baseline_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        review_id INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        predicted_sentiment TEXT NOT NULL,
        confidence_score REAL NOT NULL,
        predicted_at TIMESTAMP NOT NULL,
        FOREIGN KEY (review_id) REFERENCES reviews(id)
    )
    """
    conn.execute(create_table_query)
    
    # Xóa kết quả cũ của model này (nếu có)
    conn.execute(f"DELETE FROM baseline_predictions WHERE model_name = ?", (model_name,))
    
    # Insert kết quả mới
    insert_data = results_df[['id', 'predicted_sentiment', 'confidence_score']].copy()
    insert_data['model_name'] = model_name
    insert_data['predicted_at'] = datetime.now()
    insert_data.columns = ['review_id', 'predicted_sentiment', 'confidence_score', 
                           'model_name', 'predicted_at']
    
    insert_data.to_sql('baseline_predictions', conn, if_exists='append', index=False)
    conn.commit()
    
    print(f"   ✅ Đã lưu {len(results_df):,} dự đoán vào database!")
    
    # Export ra CSV
    print(f"\n📁 Đang xuất kết quả ra CSV...")
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'{model_name.lower()}_predictions_{timestamp}.csv'
    csv_path = output_dir / csv_filename
    
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ Đã xuất file: {csv_path}")
    
    return results_df


# ============================================================================
# HÀM LƯU MODEL
# ============================================================================

def save_model(model: Pipeline, model_name: str = MODEL_NAME):
    """
    Lưu trained model vào file.
    
    Args:
        model: Trained model pipeline
        model_name: Tên model
    """
    print(f"\n💾 Đang lưu model...")
    
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'{model_name.lower()}_{timestamp}.pkl'
    model_path = model_dir / model_filename
    
    joblib.dump(model, model_path)
    
    print(f"   ✅ Model đã được lưu: {model_path}")
    print(f"   📦 Kích thước: {model_path.stat().st_size / 1024 / 1024:.2f} MB")


# ============================================================================
# HÀM SO SÁNH VỚI MÔ HÌNH HIỆN TẠI
# ============================================================================

def compare_with_existing_model(
    baseline_results: pd.DataFrame,
    conn: sqlite3.Connection
):
    """
    So sánh kết quả của baseline model với mô hình ABSA hiện tại.
    
    Args:
        baseline_results: DataFrame kết quả của baseline model
        conn: Database connection
    """
    print("\n" + "="*70)
    print("SO SÁNH VỚI MÔ HÌNH ABSA HIỆN TẠI")
    print("="*70)
    
    # Load kết quả của mô hình ABSA (từ aspect_sentiments)
    print(f"\n📊 Đang load kết quả từ mô hình ABSA...")
    
    query = """
    SELECT 
        r.id,
        r.text,
        r.rating,
        a.sentiment,
        AVG(a.confidence_score) as avg_confidence
    FROM reviews r
    INNER JOIN aspect_sentiments a ON r.id = a.review_id
    WHERE r.id IN ({})
    GROUP BY r.id
    """.format(','.join(map(str, baseline_results['id'].tolist())))
    
    absa_results = pd.read_sql_query(query, conn)
    
    if len(absa_results) == 0:
        print(f"   ⚠️  Không tìm thấy kết quả từ mô hình ABSA")
        print(f"   💡 Đảm bảo đã chạy: python scripts/run_analysis.py --category <category>")
        return
    
    print(f"   ✅ Đã load {len(absa_results):,} reviews từ ABSA model")
    
    # Merge hai kết quả
    comparison = baseline_results[['id', 'rating', 'sentiment_label', 
                                   'predicted_sentiment', 'confidence_score']].merge(
        absa_results[['id', 'sentiment', 'avg_confidence']],
        on='id',
        how='inner'
    )
    
    comparison.columns = ['id', 'rating', 'ground_truth', 
                         'baseline_pred', 'baseline_conf',
                         'absa_pred', 'absa_conf']
    
    print(f"\n📊 So sánh trên {len(comparison):,} reviews chung:")
    
    # Tính accuracy cho từng model
    baseline_acc = (comparison['baseline_pred'] == comparison['ground_truth']).mean()
    absa_acc = (comparison['absa_pred'] == comparison['ground_truth']).mean()
    
    print(f"\n   🎯 Accuracy:")
    print(f"      • Baseline Model: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"      • ABSA Model:     {absa_acc:.4f} ({absa_acc*100:.2f}%)")
    print(f"      • Chênh lệch:     {(absa_acc - baseline_acc)*100:+.2f}%")
    
    # So sánh confidence
    print(f"\n   📊 Confidence Score:")
    print(f"      • Baseline: Mean={comparison['baseline_conf'].mean():.4f}, "
          f"Median={comparison['baseline_conf'].median():.4f}")
    print(f"      • ABSA:     Mean={comparison['absa_conf'].mean():.4f}, "
          f"Median={comparison['absa_conf'].median():.4f}")
    
    # Phân tích agreement
    agreement = (comparison['baseline_pred'] == comparison['absa_pred']).sum()
    agreement_rate = agreement / len(comparison)
    
    print(f"\n   🤝 Agreement giữa 2 models:")
    print(f"      • Số reviews đồng ý: {agreement:,} / {len(comparison):,}")
    print(f"      • Agreement rate: {agreement_rate:.4f} ({agreement_rate*100:.2f}%)")
    
    # Export comparison
    output_dir = Path(RESULTS_DIR)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_path = output_dir / f'model_comparison_{timestamp}.csv'
    comparison.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"\n   📁 Đã xuất file so sánh: {comparison_path}")


# ============================================================================
# HÀM MAIN
# ============================================================================

def main():
    """Hàm main thực thi toàn bộ pipeline."""
    print("="*70)
    print("BASELINE MODEL TRAINING - TF-IDF + LOGISTIC REGRESSION")
    print("="*70)
    print(f"\nModel Name: {MODEL_NAME}")
    print(f"Database: {DB_PATH}")
    print(f"Random State: {RANDOM_STATE}")
    print(f"Test Size: {TEST_SIZE}")
    
    try:
        # Kết nối database
        print(f"\n{'='*70}")
        print("BƯỚC 1: KẾT NỐI DATABASE")
        print(f"{'='*70}")
        
        conn = connect_db(DB_PATH)
        print(f"✅ Đã kết nối: {DB_PATH}")
        
        # Load dữ liệu
        print(f"\n{'='*70}")
        print("BƯỚC 2: LOAD DỮ LIỆU")
        print(f"{'='*70}")
        
        df = load_data_from_db(conn)
        
        if len(df) == 0:
            print(f"\n❌ Error: Không có dữ liệu để train!")
            print(f"   Đảm bảo đã load reviews vào database.")
            return 1
        
        # Train model
        print(f"\n{'='*70}")
        print("BƯỚC 3: TRAIN MODEL")
        print(f"{'='*70}")
        
        model, train_df, test_df = train_baseline_model(df, TEST_SIZE)
        
        # Dự đoán và lưu kết quả
        print(f"\n{'='*70}")
        print("BƯỚC 4: DỰ ĐOÁN VÀ LƯU KẾT QUẢ")
        print(f"{'='*70}")
        
        results_df = predict_and_save_results(model, test_df, conn, MODEL_NAME)
        
        # Lưu model
        save_model(model, MODEL_NAME)
        
        # So sánh với mô hình hiện tại
        compare_with_existing_model(results_df, conn)
        
        # Đóng kết nối
        conn.close()
        
        # Tổng kết
        print(f"\n{'='*70}")
        print("✅ HOÀN THÀNH!")
        print(f"{'='*70}")
        
        print(f"\n📊 Tóm tắt:")
        print(f"   • Model: {MODEL_NAME}")
        print(f"   • Train set: {len(train_df):,} reviews")
        print(f"   • Test set: {len(test_df):,} reviews")
        print(f"   • Kết quả đã được lưu vào database (bảng: baseline_predictions)")
        
        print(f"\n💡 Bước tiếp theo:")
        print(f"   1. Chạy script visualize để so sánh:")
        print(f"      python scripts/visualize_evaluation.py")
        print(f"   2. Hoặc import vào Power BI từ folder: {RESULTS_DIR}")
        
        print(f"\n{'='*70}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

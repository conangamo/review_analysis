"""
Script phân tích và trực quan hóa độ chính xác của AI model.

Nhiệm vụ 1: Pseudo-Ground Truth bằng Số sao (Rating)
- Phân tích mối tương quan giữa rating và sentiment dự đoán

Nhiệm vụ 2: Phân bố Độ tự tin (Confidence Score)
- Visualize phân bố confidence score theo từng sentiment

Author: AI-powered ABSA System
Date: 2026
"""

import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime


# ============================================================================
# CẤU HÌNH
# ============================================================================

DB_PATH = "data/processed/reviews.db"
OUTPUT_DIR = "data/processed/evaluation_results"

# Định nghĩa màu sắc cho các sentiment
SENTIMENT_COLORS = {
    'positive': '#2ecc71',  # Xanh lá
    'negative': '#e74c3c',  # Đỏ
    'neutral': '#95a5a6'    # Xám
}


# ============================================================================
# HÀM KẾT NỐI DATABASE
# ============================================================================

def connect_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Kết nối đến SQLite database.
    
    Args:
        db_path: Đường dẫn đến file database
        
    Returns:
        sqlite3.Connection object
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database không tồn tại: {db_path}")
    
    conn = sqlite3.connect(db_path)
    return conn


# ============================================================================
# NHIỆM VỤ 1: PSEUDO-GROUND TRUTH BẰNG SỐ SAO
# ============================================================================

def get_rating_sentiment_distribution(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Truy vấn phân bố sentiment theo rating (1-5 sao).
    
    Query JOIN hai bảng reviews và aspect_sentiments để đếm số lượng
    từng nhãn sentiment phân nhóm theo rating.
    
    Args:
        conn: Database connection
        
    Returns:
        DataFrame với cột: rating, sentiment, count
    """
    query = """
    SELECT 
        r.rating,
        a.sentiment,
        COUNT(*) as count
    FROM reviews r
    INNER JOIN aspect_sentiments a ON r.id = a.review_id
    WHERE r.rating IS NOT NULL
    GROUP BY r.rating, a.sentiment
    ORDER BY r.rating, a.sentiment
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Đảm bảo rating là kiểu số
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Làm tròn rating về số nguyên (1-5)
    df['rating'] = df['rating'].round().astype(int)
    
    return df


def plot_rating_sentiment_stacked_bar(df: pd.DataFrame) -> go.Figure:
    """
    Vẽ biểu đồ 100% Stacked Bar Chart cho phân bố sentiment theo rating.
    
    - Trục X: Rating (1-5 sao)
    - Trục Y: Phần trăm (0-100%)
    - Màu sắc: positive (xanh lá), negative (đỏ), neutral (xám)
    
    Args:
        df: DataFrame từ get_rating_sentiment_distribution()
        
    Returns:
        plotly Figure object
    """
    # Tính phần trăm cho mỗi nhóm rating
    df_percent = df.copy()
    df_percent['total'] = df_percent.groupby('rating')['count'].transform('sum')
    df_percent['percentage'] = (df_percent['count'] / df_percent['total'] * 100).round(2)
    
    # Tạo biểu đồ 100% stacked bar
    fig = px.bar(
        df_percent,
        x='rating',
        y='percentage',
        color='sentiment',
        title='Phân bố Sentiment theo Rating (Pseudo-Ground Truth)',
        labels={
            'rating': 'Số sao (Rating)',
            'percentage': 'Phần trăm (%)',
            'sentiment': 'Sentiment'
        },
        color_discrete_map=SENTIMENT_COLORS,
        text='percentage',
        category_orders={'sentiment': ['positive', 'neutral', 'negative']}
    )
    
    # Định dạng text hiển thị trên bar
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='inside',
        textfont_size=11
    )
    
    # Cập nhật layout
    fig.update_layout(
        barmode='stack',
        xaxis=dict(
            title='Số sao (Rating)',
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        yaxis=dict(
            title='Phần trăm (%)',
            range=[0, 100]
        ),
        legend=dict(
            title='Sentiment',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=500,
        hovermode='x unified'
    )
    
    return fig


# ============================================================================
# NHIỆM VỤ 2: PHÂN BỐ ĐỘ TỰ TIN (CONFIDENCE SCORE)
# ============================================================================

def get_confidence_distribution(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Truy vấn phân bố confidence score theo sentiment.
    
    Args:
        conn: Database connection
        
    Returns:
        DataFrame với cột: confidence_score, sentiment
    """
    query = """
    SELECT 
        confidence_score,
        sentiment
    FROM aspect_sentiments
    WHERE confidence_score IS NOT NULL
    ORDER BY confidence_score
    """
    
    df = pd.read_sql_query(query, conn)
    return df


def plot_confidence_histogram(df: pd.DataFrame, bins: int = 50) -> go.Figure:
    """
    Vẽ biểu đồ Histogram cho phân bố confidence score.
    
    - Trục X: Confidence Score (0.0 - 1.0)
    - Trục Y: Số lượng (Count)
    - Màu sắc: Theo sentiment (positive, negative, neutral)
    
    Args:
        df: DataFrame từ get_confidence_distribution()
        bins: Số lượng bins cho histogram (mặc định 50)
        
    Returns:
        plotly Figure object
    """
    fig = px.histogram(
        df,
        x='confidence_score',
        color='sentiment',
        nbins=bins,
        title='Phân bố Độ tự tin (Confidence Score) của Model',
        labels={
            'confidence_score': 'Confidence Score',
            'count': 'Số lượng',
            'sentiment': 'Sentiment'
        },
        color_discrete_map=SENTIMENT_COLORS,
        barmode='overlay',
        opacity=0.7,
        category_orders={'sentiment': ['positive', 'neutral', 'negative']}
    )
    
    # Cập nhật layout
    fig.update_layout(
        xaxis=dict(
            title='Confidence Score',
            range=[0, 1],
            tickformat='.2f'
        ),
        yaxis=dict(
            title='Số lượng'
        ),
        legend=dict(
            title='Sentiment',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=500,
        hovermode='x'
    )
    
    return fig


# ============================================================================
# HÀM XUẤT DỮ LIỆU RA CSV CHO POWER BI
# ============================================================================

def export_to_csv(
    df_rating: pd.DataFrame,
    df_confidence: pd.DataFrame,
    output_dir: str = OUTPUT_DIR
) -> tuple[str, str]:
    """
    Xuất 2 dataset ra file CSV để import vào Power BI.
    
    Args:
        df_rating: DataFrame phân bố rating-sentiment
        df_confidence: DataFrame phân bố confidence score
        output_dir: Thư mục lưu file CSV
        
    Returns:
        Tuple (path_rating_csv, path_confidence_csv)
    """
    # Tạo thư mục output nếu chưa tồn tại
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Tạo timestamp cho tên file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Export dataset 1: Rating-Sentiment Distribution
    rating_csv = output_path / f'rating_sentiment_distribution_{timestamp}.csv'
    df_rating.to_csv(rating_csv, index=False, encoding='utf-8-sig')
    
    # Export dataset 2: Confidence Score Distribution
    confidence_csv = output_path / f'confidence_score_distribution_{timestamp}.csv'
    df_confidence.to_csv(confidence_csv, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Đã xuất file CSV:")
    print(f"   📊 Rating-Sentiment: {rating_csv}")
    print(f"   📊 Confidence Score: {confidence_csv}")
    
    return str(rating_csv), str(confidence_csv)


# ============================================================================
# HÀM WRAPPER ĐỂ TÍCH HỢP VỚI STREAMLIT
# ============================================================================

def create_rating_sentiment_chart(db_path: str = DB_PATH) -> go.Figure:
    """
    Wrapper function để tạo biểu đồ rating-sentiment.
    Dễ dàng gọi từ Streamlit: st.plotly_chart(create_rating_sentiment_chart())
    
    Args:
        db_path: Đường dẫn đến database
        
    Returns:
        plotly Figure object
    """
    conn = connect_db(db_path)
    df = get_rating_sentiment_distribution(conn)
    fig = plot_rating_sentiment_stacked_bar(df)
    conn.close()
    return fig


def create_confidence_histogram_chart(
    db_path: str = DB_PATH,
    bins: int = 50
) -> go.Figure:
    """
    Wrapper function để tạo biểu đồ confidence histogram.
    Dễ dàng gọi từ Streamlit: st.plotly_chart(create_confidence_histogram_chart())
    
    Args:
        db_path: Đường dẫn đến database
        bins: Số lượng bins cho histogram
        
    Returns:
        plotly Figure object
    """
    conn = connect_db(db_path)
    df = get_confidence_distribution(conn)
    fig = plot_confidence_histogram(df, bins)
    conn.close()
    return fig


# ============================================================================
# HÀM MAIN - THỰC THI TẤT CẢ NHIỆM VỤ
# ============================================================================

def main():
    """
    Hàm main thực thi tất cả các nhiệm vụ:
    1. Vẽ biểu đồ Rating-Sentiment
    2. Vẽ biểu đồ Confidence Distribution
    3. So sánh Baseline vs ABSA Model (nếu có)
    4. Xuất dữ liệu ra CSV cho Power BI
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phân tích và trực quan hóa độ chính xác của AI model"
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='So sánh Baseline Model với ABSA Model'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHÂN TÍCH ĐỘ CHÍNH XÁC MODEL AI - ABSA SYSTEM")
    print("=" * 70)
    
    # Kết nối database
    print(f"\n🔗 Kết nối database: {DB_PATH}")
    conn = connect_db(DB_PATH)
    
    # ========================================================================
    # NHIỆM VỤ 1: RATING-SENTIMENT DISTRIBUTION
    # ========================================================================
    print("\n" + "=" * 70)
    print("NHIỆM VỤ 1: PSEUDO-GROUND TRUTH BẰNG SỐ SAO")
    print("=" * 70)
    
    print("\n📊 Đang truy vấn dữ liệu rating-sentiment...")
    df_rating = get_rating_sentiment_distribution(conn)
    print(f"   ✅ Lấy được {len(df_rating)} dòng dữ liệu")
    print(f"\n   Preview data:")
    print(df_rating.head(10).to_string(index=False))
    
    print("\n📈 Đang vẽ biểu đồ 100% Stacked Bar Chart...")
    fig_rating = plot_rating_sentiment_stacked_bar(df_rating)
    print("   ✅ Hoàn thành!")
    
    # Hiển thị biểu đồ
    fig_rating.show()
    
    # ========================================================================
    # NHIỆM VỤ 2: CONFIDENCE SCORE DISTRIBUTION
    # ========================================================================
    print("\n" + "=" * 70)
    print("NHIỆM VỤ 2: PHÂN BỐ ĐỘ TỰ TIN (CONFIDENCE SCORE)")
    print("=" * 70)
    
    print("\n📊 Đang truy vấn dữ liệu confidence score...")
    df_confidence = get_confidence_distribution(conn)
    print(f"   ✅ Lấy được {len(df_confidence)} dòng dữ liệu")
    print(f"\n   Thống kê mô tả:")
    print(df_confidence['confidence_score'].describe().to_string())
    
    print("\n📈 Đang vẽ biểu đồ Histogram...")
    fig_confidence = plot_confidence_histogram(df_confidence, bins=50)
    print("   ✅ Hoàn thành!")
    
    # Hiển thị biểu đồ
    fig_confidence.show()
    
    # ========================================================================
    # XUẤT DỮ LIỆU RA CSV
    # ========================================================================
    print("\n" + "=" * 70)
    print("XUẤT DỮ LIỆU CHO POWER BI")
    print("=" * 70)
    
    rating_csv, confidence_csv = export_to_csv(df_rating, df_confidence)
    
    # ========================================================================
    # NHIỆM VỤ 3: SO SÁNH BASELINE VS ABSA (NẾU ĐƯỢC YÊU CẦU)
    # ========================================================================
    if args.compare:
        print("\n" + "=" * 70)
        print("NHIỆM VỤ 3: SO SÁNH BASELINE VS ABSA MODEL")
        print("=" * 70)
        
        create_model_comparison_dashboard(DB_PATH)
    
    # Đóng kết nối
    conn.close()
    
    # ========================================================================
    # THỐNG KÊ TỔNG KẾT
    # ========================================================================
    print("\n" + "=" * 70)
    print("THỐNG KÊ TỔNG KẾT")
    print("=" * 70)
    
    # Thống kê từ dataset rating
    print("\n📊 Phân bố Sentiment theo Rating:")
    summary_rating = df_rating.groupby('sentiment')['count'].sum().sort_values(ascending=False)
    total_sentiments = summary_rating.sum()
    for sentiment, count in summary_rating.items():
        percentage = (count / total_sentiments * 100)
        print(f"   • {sentiment.capitalize():8s}: {count:6,d} ({percentage:5.2f}%)")
    
    # Thống kê từ dataset confidence
    print(f"\n📊 Thống kê Confidence Score:")
    print(f"   • Trung bình: {df_confidence['confidence_score'].mean():.4f}")
    print(f"   • Trung vị:   {df_confidence['confidence_score'].median():.4f}")
    print(f"   • Độ lệch chuẩn: {df_confidence['confidence_score'].std():.4f}")
    print(f"   • Min: {df_confidence['confidence_score'].min():.4f}")
    print(f"   • Max: {df_confidence['confidence_score'].max():.4f}")
    
    # Phân bố confidence theo sentiment
    print(f"\n📊 Confidence Score theo Sentiment:")
    for sentiment in ['positive', 'negative', 'neutral']:
        sentiment_data = df_confidence[df_confidence['sentiment'] == sentiment]['confidence_score']
        if len(sentiment_data) > 0:
            print(f"   • {sentiment.capitalize():8s}: "
                  f"Mean={sentiment_data.mean():.4f}, "
                  f"Median={sentiment_data.median():.4f}")
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH TẤT CẢ NHIỆM VỤ!")
    print("=" * 70)
    print("\n💡 Cách sử dụng:")
    print("   • Xem biểu đồ cơ bản:")
    print("     python scripts/visualize_evaluation.py")
    print("   • So sánh với Baseline Model:")
    print("     python scripts/visualize_evaluation.py --compare")
    print("\n💡 Với Streamlit:")
    print("   import streamlit as st")
    print("   from scripts.visualize_evaluation import create_rating_sentiment_chart")
    print("   ")
    print("   fig = create_rating_sentiment_chart()")
    print("   st.plotly_chart(fig)")
    print("\n💡 File CSV đã được xuất, có thể import vào Power BI!")
    print("=" * 70)


# ============================================================================
# NHIỆM VỤ 3: SO SÁNH BASELINE MODEL VỚI ABSA MODEL
# ============================================================================

def get_model_comparison_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Truy vấn và so sánh kết quả giữa Baseline Model và ABSA Model.
    
    Args:
        conn: Database connection
        
    Returns:
        DataFrame với cột: review_id, rating, baseline_pred, baseline_conf, 
                          absa_pred, absa_conf
    """
    query = """
    SELECT 
        r.id as review_id,
        r.rating,
        bp.predicted_sentiment as baseline_pred,
        bp.confidence_score as baseline_conf,
        a.sentiment as absa_pred,
        AVG(a.confidence_score) as absa_conf
    FROM reviews r
    LEFT JOIN baseline_predictions bp ON r.id = bp.review_id 
        AND bp.model_name = 'Baseline_ML_Model'
    LEFT JOIN aspect_sentiments a ON r.id = a.review_id
    WHERE bp.predicted_sentiment IS NOT NULL 
        AND a.sentiment IS NOT NULL
        AND r.rating IS NOT NULL
    GROUP BY r.id, bp.predicted_sentiment, bp.confidence_score
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Tạo ground truth từ rating
    def rating_to_sentiment(rating: float) -> str:
        if rating <= 2.0:
            return 'negative'
        elif rating >= 4.0:
            return 'positive'
        else:
            return 'neutral'
    
    df['ground_truth'] = df['rating'].apply(rating_to_sentiment)
    
    # Tính correct/incorrect cho từng model
    df['baseline_correct'] = (df['baseline_pred'] == df['ground_truth']).astype(int)
    df['absa_correct'] = (df['absa_pred'] == df['ground_truth']).astype(int)
    
    return df


def plot_model_accuracy_comparison(df: pd.DataFrame) -> go.Figure:
    """
    Vẽ biểu đồ so sánh accuracy giữa Baseline và ABSA model.
    
    Args:
        df: DataFrame từ get_model_comparison_data()
        
    Returns:
        plotly Figure object
    """
    # Tính accuracy cho từng model
    baseline_acc = df['baseline_correct'].mean() * 100
    absa_acc = df['absa_correct'].mean() * 100
    
    # Tạo DataFrame cho biểu đồ
    comparison_data = pd.DataFrame({
        'Model': ['Baseline (TF-IDF+LR)', 'ABSA (Zero-shot)'],
        'Accuracy': [baseline_acc, absa_acc]
    })
    
    # Tạo bar chart
    fig = px.bar(
        comparison_data,
        x='Model',
        y='Accuracy',
        title='So sánh Accuracy: Baseline vs ABSA Model',
        labels={'Accuracy': 'Accuracy (%)'},
        text='Accuracy',
        color='Model',
        color_discrete_sequence=['#3498db', '#e67e22']
    )
    
    # Định dạng
    fig.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside',
        textfont_size=14
    )
    
    fig.update_layout(
        yaxis=dict(
            title='Accuracy (%)',
            range=[0, 100]
        ),
        showlegend=False,
        height=500
    )
    
    return fig


def plot_confidence_comparison_boxplot(df: pd.DataFrame) -> go.Figure:
    """
    Vẽ boxplot so sánh confidence score giữa 2 models.
    
    Args:
        df: DataFrame từ get_model_comparison_data()
        
    Returns:
        plotly Figure object
    """
    # Chuẩn bị dữ liệu
    baseline_data = pd.DataFrame({
        'Model': 'Baseline (TF-IDF+LR)',
        'Confidence': df['baseline_conf']
    })
    
    absa_data = pd.DataFrame({
        'Model': 'ABSA (Zero-shot)',
        'Confidence': df['absa_conf']
    })
    
    combined_data = pd.concat([baseline_data, absa_data], ignore_index=True)
    
    # Tạo boxplot
    fig = px.box(
        combined_data,
        x='Model',
        y='Confidence',
        title='So sánh Confidence Score: Baseline vs ABSA Model',
        labels={'Confidence': 'Confidence Score'},
        color='Model',
        color_discrete_sequence=['#3498db', '#e67e22']
    )
    
    fig.update_layout(
        yaxis=dict(
            title='Confidence Score',
            range=[0, 1]
        ),
        showlegend=False,
        height=500
    )
    
    return fig


def plot_per_sentiment_accuracy(df: pd.DataFrame) -> go.Figure:
    """
    Vẽ biểu đồ so sánh accuracy theo từng sentiment class.
    
    Args:
        df: DataFrame từ get_model_comparison_data()
        
    Returns:
        plotly Figure object
    """
    # Tính accuracy cho từng sentiment
    accuracy_data = []
    
    for sentiment in ['positive', 'neutral', 'negative']:
        sentiment_df = df[df['ground_truth'] == sentiment]
        
        if len(sentiment_df) > 0:
            baseline_acc = sentiment_df['baseline_correct'].mean() * 100
            absa_acc = sentiment_df['absa_correct'].mean() * 100
            
            accuracy_data.append({
                'Sentiment': sentiment.capitalize(),
                'Model': 'Baseline',
                'Accuracy': baseline_acc
            })
            
            accuracy_data.append({
                'Sentiment': sentiment.capitalize(),
                'Model': 'ABSA',
                'Accuracy': absa_acc
            })
    
    accuracy_df = pd.DataFrame(accuracy_data)
    
    # Tạo grouped bar chart
    fig = px.bar(
        accuracy_df,
        x='Sentiment',
        y='Accuracy',
        color='Model',
        barmode='group',
        title='Accuracy theo Sentiment: Baseline vs ABSA',
        labels={'Accuracy': 'Accuracy (%)'},
        text='Accuracy',
        color_discrete_sequence=['#3498db', '#e67e22']
    )
    
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        textfont_size=11
    )
    
    fig.update_layout(
        yaxis=dict(
            title='Accuracy (%)',
            range=[0, 100]
        ),
        legend=dict(
            title='Model',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=500
    )
    
    return fig


def export_comparison_summary(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """
    Xuất bảng tóm tắt so sánh ra CSV.
    
    Args:
        df: DataFrame từ get_model_comparison_data()
        output_dir: Thư mục output
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Tính toán metrics tổng quát
    summary_data = {
        'Metric': [],
        'Baseline (TF-IDF+LR)': [],
        'ABSA (Zero-shot)': []
    }
    
    # Overall accuracy
    summary_data['Metric'].append('Overall Accuracy (%)')
    summary_data['Baseline (TF-IDF+LR)'].append(f"{df['baseline_correct'].mean() * 100:.2f}")
    summary_data['ABSA (Zero-shot)'].append(f"{df['absa_correct'].mean() * 100:.2f}")
    
    # Confidence score mean
    summary_data['Metric'].append('Mean Confidence')
    summary_data['Baseline (TF-IDF+LR)'].append(f"{df['baseline_conf'].mean():.4f}")
    summary_data['ABSA (Zero-shot)'].append(f"{df['absa_conf'].mean():.4f}")
    
    # Confidence score median
    summary_data['Metric'].append('Median Confidence')
    summary_data['Baseline (TF-IDF+LR)'].append(f"{df['baseline_conf'].median():.4f}")
    summary_data['ABSA (Zero-shot)'].append(f"{df['absa_conf'].median():.4f}")
    
    # Per-sentiment accuracy
    for sentiment in ['positive', 'neutral', 'negative']:
        sentiment_df = df[df['ground_truth'] == sentiment]
        if len(sentiment_df) > 0:
            summary_data['Metric'].append(f'{sentiment.capitalize()} Accuracy (%)')
            summary_data['Baseline (TF-IDF+LR)'].append(
                f"{sentiment_df['baseline_correct'].mean() * 100:.2f}"
            )
            summary_data['ABSA (Zero-shot)'].append(
                f"{sentiment_df['absa_correct'].mean() * 100:.2f}"
            )
    
    # Agreement rate
    agreement = (df['baseline_pred'] == df['absa_pred']).sum()
    agreement_rate = agreement / len(df) * 100
    summary_data['Metric'].append('Agreement Rate (%)')
    summary_data['Baseline (TF-IDF+LR)'].append(f"{agreement_rate:.2f}")
    summary_data['ABSA (Zero-shot)'].append(f"{agreement_rate:.2f}")
    
    summary_df = pd.DataFrame(summary_data)
    
    # Export
    summary_file = output_path / f'model_comparison_summary_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Đã xuất summary comparison:")
    print(f"   📊 {summary_file}")


def create_model_comparison_dashboard(db_path: str = DB_PATH):
    """
    Tạo dashboard so sánh Baseline vs ABSA model.
    
    Args:
        db_path: Đường dẫn đến database
    """
    print("\n" + "="*70)
    print("SO SÁNH BASELINE MODEL vs ABSA MODEL")
    print("="*70)
    
    conn = connect_db(db_path)
    
    print(f"\n📊 Đang load dữ liệu so sánh...")
    df = get_model_comparison_data(conn)
    
    if len(df) == 0:
        print(f"\n⚠️  Không tìm thấy dữ liệu để so sánh!")
        print(f"   Đảm bảo đã chạy:")
        print(f"   1. python scripts/train_baseline.py")
        print(f"   2. python scripts/run_analysis.py --category <category>")
        conn.close()
        return
    
    print(f"   ✅ Đã load {len(df):,} reviews để so sánh")
    
    # Tính toán metrics
    baseline_acc = df['baseline_correct'].mean() * 100
    absa_acc = df['absa_correct'].mean() * 100
    agreement = (df['baseline_pred'] == df['absa_pred']).sum()
    agreement_rate = agreement / len(df) * 100
    
    print(f"\n📈 Kết quả so sánh:")
    print(f"   • Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"   • ABSA Accuracy:     {absa_acc:.2f}%")
    print(f"   • Chênh lệch:        {(absa_acc - baseline_acc):+.2f}%")
    print(f"   • Agreement:         {agreement_rate:.2f}%")
    
    # Vẽ các biểu đồ
    print(f"\n📊 Đang tạo biểu đồ so sánh...")
    
    print(f"   1. Accuracy Comparison...")
    fig_acc = plot_model_accuracy_comparison(df)
    fig_acc.show()
    
    print(f"   2. Confidence Comparison...")
    fig_conf = plot_confidence_comparison_boxplot(df)
    fig_conf.show()
    
    print(f"   3. Per-Sentiment Accuracy...")
    fig_sentiment = plot_per_sentiment_accuracy(df)
    fig_sentiment.show()
    
    # Export summary
    export_comparison_summary(df)
    
    conn.close()
    
    print(f"\n✅ Hoàn thành!")


if __name__ == "__main__":
    main()

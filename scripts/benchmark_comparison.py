"""
Script benchmark so sánh hiệu suất giữa hai mô hình Zero-shot Classification.

Mô hình 1 (Hiện tại): valhalla/distilbart-mnli-12-3
Mô hình 2 (Baseline): facebook/bart-large-mnli

Đo đạc các chỉ số:
- Thời gian xử lý (reviews/giây)
- Pseudo-Accuracy (tương thích với rating)
- VRAM usage (ước tính)

Author: ABSA Benchmark System
Date: 2026
"""

import sqlite3
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import transformers
from transformers import pipeline
import torch

# Import từ dự án
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.aspect_manager import AspectManager
from src.core.config_loader import ConfigLoader


# ============================================================================
# CẤU HÌNH
# ============================================================================

DB_PATH = "data/processed/reviews.db"
OUTPUT_DIR = "data/processed/benchmark_results"
SAMPLE_SIZE = 50  # Số reviews để test

# Định nghĩa 2 model cần so sánh
MODEL_1 = {
    'name': 'DistilBART-MNLI',
    'full_name': 'valhalla/distilbart-mnli-12-3',
    'description': 'Mô hình hiện tại của dự án (nhẹ, nhanh)'
}

MODEL_2 = {
    'name': 'BART-Large-MNLI', 
    'full_name': 'facebook/bart-large-mnli',
    'description': 'Mô hình baseline (lớn, chính xác)'
}

# Các aspect cơ bản để test
TEST_ASPECTS = ['battery', 'screen', 'camera', 'performance', 'price', 'design']


# ============================================================================
# HÀM KẾT NỐI DATABASE VÀ LẤY MẪU
# ============================================================================

def connect_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Kết nối database SQLite."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database không tồn tại: {db_path}")
    return sqlite3.connect(db_path)


def get_sample_reviews(conn: sqlite3.Connection, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Lấy ngẫu nhiên sample_size reviews từ database.
    
    Args:
        conn: Database connection
        sample_size: Số lượng reviews cần lấy
        
    Returns:
        DataFrame với cột: id, text, rating
    """
    query = f"""
    SELECT 
        id,
        text,
        rating
    FROM reviews
    WHERE text IS NOT NULL 
        AND LENGTH(text) >= 50
        AND LENGTH(text) <= 500
        AND rating IS NOT NULL
    ORDER BY RANDOM()
    LIMIT {sample_size}
    """
    
    df = pd.read_sql_query(query, conn)
    print(f"📊 Đã lấy {len(df)} reviews để test")
    return df


# ============================================================================
# HÀM KHỞI TẠO MODEL
# ============================================================================

def initialize_model(model_name: str, device: str = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Khởi tạo Zero-shot Classification model.
    
    Args:
        model_name: Tên model từ Hugging Face
        device: Device để chạy (cuda/cpu), None = auto-detect
        
    Returns:
        Tuple (classifier, model_info)
    """
    print(f"\n🤖 Đang khởi tạo model: {model_name}")
    
    # Auto-detect device nếu không được chỉ định
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Đo VRAM trước khi load model
    vram_before = 0
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / 1024**2  # MB
    
    # Khởi tạo model
    start_time = time.time()
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0 if device == "cuda" else -1
    )
    load_time = time.time() - start_time
    
    # Đo VRAM sau khi load model
    vram_after = 0
    vram_used = 0
    if device == "cuda" and torch.cuda.is_available():
        vram_after = torch.cuda.memory_allocated() / 1024**2  # MB
        vram_used = vram_after - vram_before
    
    model_info = {
        'device': device,
        'load_time': load_time,
        'vram_mb': vram_used,
        'has_gpu': torch.cuda.is_available()
    }
    
    print(f"   ✅ Đã load xong trong {load_time:.2f}s")
    if device == "cuda":
        print(f"   💾 VRAM sử dụng: {vram_used:.1f} MB")
    
    return classifier, model_info


# ============================================================================
# HÀM PHÂN TÍCH SENTIMENT
# ============================================================================

def analyze_sentiment_simple(
    classifier: Any,
    text: str,
    aspect: str = None
) -> Dict[str, Any]:
    """
    Phân tích sentiment cho một review (đơn giản, không dùng AspectManager).
    
    Args:
        classifier: Zero-shot classifier
        text: Review text
        aspect: Tên aspect (nếu có)
        
    Returns:
        Dict với sentiment và confidence
    """
    candidate_labels = ['positive', 'negative', 'neutral']
    
    # Tạo hypothesis template
    if aspect:
        hypothesis_template = f"This review about {aspect} is {{}}"
    else:
        hypothesis_template = "This review is {}"
    
    # Classify
    result = classifier(
        text,
        candidate_labels=candidate_labels,
        hypothesis_template=hypothesis_template,
        multi_label=False
    )
    
    return {
        'sentiment': result['labels'][0],
        'confidence': result['scores'][0],
        'all_scores': {
            label: score 
            for label, score in zip(result['labels'], result['scores'])
        }
    }


def detect_aspects_in_text(text: str, aspects: List[str]) -> List[str]:
    """
    Detect aspects xuất hiện trong text (simple keyword matching).
    
    Args:
        text: Review text
        aspects: List các aspect cần check
        
    Returns:
        List các aspect được tìm thấy
    """
    text_lower = text.lower()
    detected = []
    
    # Keyword mapping cho từng aspect
    aspect_keywords = {
        'battery': ['battery', 'charge', 'charging', 'power'],
        'screen': ['screen', 'display', 'brightness'],
        'camera': ['camera', 'photo', 'picture', 'video'],
        'performance': ['performance', 'speed', 'fast', 'slow', 'lag'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth'],
        'design': ['design', 'look', 'style', 'build quality']
    }
    
    for aspect in aspects:
        keywords = aspect_keywords.get(aspect, [aspect])
        for keyword in keywords:
            if keyword in text_lower:
                detected.append(aspect)
                break
    
    return detected


def benchmark_model(
    classifier: Any,
    reviews_df: pd.DataFrame,
    model_name: str,
    aspects: List[str] = TEST_ASPECTS
) -> Dict[str, Any]:
    """
    Chạy benchmark cho một model.
    
    Args:
        classifier: Zero-shot classifier
        reviews_df: DataFrame chứa reviews
        model_name: Tên model (để hiển thị)
        aspects: List các aspect cần phân tích
        
    Returns:
        Dict chứa kết quả benchmark
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {model_name}")
    print(f"{'='*70}")
    
    results = []
    total_time = 0
    total_analyses = 0
    
    print(f"\n📝 Đang phân tích {len(reviews_df)} reviews...")
    
    for idx, row in reviews_df.iterrows():
        review_id = row['id']
        text = row['text']
        rating = row['rating']
        
        # Detect aspects trong review này
        detected_aspects = detect_aspects_in_text(text, aspects)
        
        if not detected_aspects:
            # Nếu không detect aspect nào, dùng overall sentiment
            detected_aspects = [None]
        
        # Phân tích sentiment cho từng aspect
        review_results = {
            'review_id': review_id,
            'text': text[:100] + '...' if len(text) > 100 else text,
            'rating': rating,
            'aspects': []
        }
        
        for aspect in detected_aspects:
            start_time = time.time()
            
            sentiment_result = analyze_sentiment_simple(
                classifier,
                text,
                aspect=aspect
            )
            
            elapsed = time.time() - start_time
            total_time += elapsed
            total_analyses += 1
            
            review_results['aspects'].append({
                'aspect': aspect if aspect else 'overall',
                'sentiment': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence']
            })
        
        results.append(review_results)
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"   ⏳ Đã xử lý {idx + 1}/{len(reviews_df)} reviews...")
    
    # Tính toán metrics
    avg_time_per_analysis = total_time / total_analyses if total_analyses > 0 else 0
    reviews_per_second = total_analyses / total_time if total_time > 0 else 0
    
    # Tính Pseudo-Accuracy (so với rating)
    pseudo_accuracy = calculate_pseudo_accuracy(results)
    
    print(f"\n✅ Hoàn thành!")
    print(f"   • Tổng thời gian: {total_time:.2f}s")
    print(f"   • Số phân tích: {total_analyses}")
    print(f"   • Tốc độ: {reviews_per_second:.2f} analyses/giây")
    print(f"   • Thời gian TB mỗi analysis: {avg_time_per_analysis*1000:.1f}ms")
    print(f"   • Pseudo-Accuracy: {pseudo_accuracy:.2%}")
    
    return {
        'model_name': model_name,
        'results': results,
        'total_time': total_time,
        'total_analyses': total_analyses,
        'avg_time_per_analysis': avg_time_per_analysis,
        'analyses_per_second': reviews_per_second,
        'pseudo_accuracy': pseudo_accuracy
    }


def calculate_pseudo_accuracy(results: List[Dict]) -> float:
    """
    Tính Pseudo-Accuracy dựa trên logic:
    - Rating 1-2: Nên là Negative
    - Rating 4-5: Nên là Positive
    - Rating 3: Neutral
    
    Args:
        results: List kết quả phân tích
        
    Returns:
        Tỷ lệ accuracy (0.0 - 1.0)
    """
    correct = 0
    total = 0
    
    for result in results:
        rating = result['rating']
        
        # Xác định expected sentiment dựa trên rating
        if rating <= 2.0:
            expected = 'negative'
        elif rating >= 4.0:
            expected = 'positive'
        else:
            expected = 'neutral'
        
        # Check sentiment của các aspects
        for aspect_result in result['aspects']:
            predicted = aspect_result['sentiment']
            
            if predicted == expected:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


# ============================================================================
# HÀM SO SÁNH VÀ XUẤT KẾT QUẢ
# ============================================================================

def compare_models(
    benchmark1: Dict[str, Any],
    benchmark2: Dict[str, Any]
) -> pd.DataFrame:
    """
    So sánh kết quả của 2 models.
    
    Args:
        benchmark1: Kết quả benchmark model 1
        benchmark2: Kết quả benchmark model 2
        
    Returns:
        DataFrame so sánh
    """
    comparison_data = {
        'Metric': [
            'Tổng thời gian (giây)',
            'Số phân tích',
            'Tốc độ (analyses/giây)',
            'Thời gian TB/analysis (ms)',
            'Pseudo-Accuracy (%)'
        ],
        benchmark1['model_name']: [
            f"{benchmark1['total_time']:.2f}",
            f"{benchmark1['total_analyses']}",
            f"{benchmark1['analyses_per_second']:.2f}",
            f"{benchmark1['avg_time_per_analysis']*1000:.1f}",
            f"{benchmark1['pseudo_accuracy']*100:.2f}"
        ],
        benchmark2['model_name']: [
            f"{benchmark2['total_time']:.2f}",
            f"{benchmark2['total_analyses']}",
            f"{benchmark2['analyses_per_second']:.2f}",
            f"{benchmark2['avg_time_per_analysis']*1000:.1f}",
            f"{benchmark2['pseudo_accuracy']*100:.2f}"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    return df


def export_results(
    benchmark1: Dict[str, Any],
    benchmark2: Dict[str, Any],
    model1_info: Dict[str, Any],
    model2_info: Dict[str, Any],
    output_dir: str = OUTPUT_DIR
):
    """
    Xuất kết quả benchmark ra file CSV.
    
    Args:
        benchmark1: Kết quả benchmark model 1
        benchmark2: Kết quả benchmark model 2
        model1_info: Thông tin model 1
        model2_info: Thông tin model 2
        output_dir: Thư mục output
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Export summary comparison
    comparison_df = compare_models(benchmark1, benchmark2)
    summary_file = output_path / f'benchmark_results_{timestamp}.csv'
    comparison_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    
    # 2. Export detailed results
    detailed_data = []
    
    for bench, model_info, model_name in [
        (benchmark1, model1_info, MODEL_1['name']),
        (benchmark2, model2_info, MODEL_2['name'])
    ]:
        for result in bench['results']:
            for aspect in result['aspects']:
                detailed_data.append({
                    'model': model_name,
                    'review_id': result['review_id'],
                    'rating': result['rating'],
                    'aspect': aspect['aspect'],
                    'sentiment': aspect['sentiment'],
                    'confidence': aspect['confidence'],
                    'text_preview': result['text']
                })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_file = output_path / f'benchmark_detailed_{timestamp}.csv'
    detailed_df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
    
    # 3. Export model info
    model_info_data = {
        'Model': [MODEL_1['name'], MODEL_2['name']],
        'Full Name': [MODEL_1['full_name'], MODEL_2['full_name']],
        'Description': [MODEL_1['description'], MODEL_2['description']],
        'Device': [model1_info['device'], model2_info['device']],
        'Load Time (s)': [f"{model1_info['load_time']:.2f}", f"{model2_info['load_time']:.2f}"],
        'VRAM (MB)': [f"{model1_info['vram_mb']:.1f}", f"{model2_info['vram_mb']:.1f}"]
    }
    
    model_info_df = pd.DataFrame(model_info_data)
    model_info_file = output_path / f'benchmark_model_info_{timestamp}.csv'
    model_info_df.to_csv(model_info_file, index=False, encoding='utf-8-sig')
    
    print(f"\n📊 Đã xuất kết quả:")
    print(f"   • Summary: {summary_file}")
    print(f"   • Detailed: {detailed_file}")
    print(f"   • Model Info: {model_info_file}")


# ============================================================================
# HÀM MAIN
# ============================================================================

def main():
    """Hàm main thực thi benchmark."""
    print("=" * 70)
    print("BENCHMARK SO SÁNH MÔ HÌNH ZERO-SHOT CLASSIFICATION")
    print("=" * 70)
    print(f"\n📌 Model 1: {MODEL_1['name']} ({MODEL_1['full_name']})")
    print(f"   → {MODEL_1['description']}")
    print(f"\n📌 Model 2: {MODEL_2['name']} ({MODEL_2['full_name']})")
    print(f"   → {MODEL_2['description']}")
    print(f"\n📊 Sample size: {SAMPLE_SIZE} reviews")
    print(f"🎯 Aspects: {', '.join(TEST_ASPECTS)}")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n🚀 GPU detected: {gpu_name}")
    else:
        print(f"\n⚠️  GPU not available, using CPU")
    
    # Kết nối database và lấy sample
    print(f"\n{'='*70}")
    print("CHUẨN BỊ DỮ LIỆU")
    print(f"{'='*70}")
    
    conn = connect_db(DB_PATH)
    reviews_df = get_sample_reviews(conn, SAMPLE_SIZE)
    conn.close()
    
    print(f"\n📈 Thống kê mẫu:")
    print(f"   • Tổng số reviews: {len(reviews_df)}")
    print(f"   • Rating trung bình: {reviews_df['rating'].mean():.2f}")
    print(f"   • Độ dài text TB: {reviews_df['text'].str.len().mean():.0f} ký tự")
    
    # Khởi tạo và benchmark Model 1
    print(f"\n{'='*70}")
    print(f"MODEL 1: {MODEL_1['name']}")
    print(f"{'='*70}")
    
    classifier1, model1_info = initialize_model(MODEL_1['full_name'])
    benchmark1 = benchmark_model(
        classifier1,
        reviews_df,
        MODEL_1['name'],
        TEST_ASPECTS
    )
    
    # Giải phóng memory
    del classifier1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Khởi tạo và benchmark Model 2
    print(f"\n{'='*70}")
    print(f"MODEL 2: {MODEL_2['name']}")
    print(f"{'='*70}")
    
    classifier2, model2_info = initialize_model(MODEL_2['full_name'])
    benchmark2 = benchmark_model(
        classifier2,
        reviews_df,
        MODEL_2['name'],
        TEST_ASPECTS
    )
    
    # Giải phóng memory
    del classifier2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # So sánh kết quả
    print(f"\n{'='*70}")
    print("KẾT QUẢ SO SÁNH")
    print(f"{'='*70}")
    
    comparison_df = compare_models(benchmark1, benchmark2)
    print(f"\n{comparison_df.to_string(index=False)}")
    
    # Phân tích chi tiết
    print(f"\n{'='*70}")
    print("PHÂN TÍCH CHI TIẾT")
    print(f"{'='*70}")
    
    # Tốc độ
    speedup = benchmark1['analyses_per_second'] / benchmark2['analyses_per_second']
    print(f"\n🚀 Tốc độ:")
    print(f"   • {MODEL_1['name']} nhanh hơn {speedup:.2f}x so với {MODEL_2['name']}")
    
    # Accuracy
    acc_diff = (benchmark1['pseudo_accuracy'] - benchmark2['pseudo_accuracy']) * 100
    print(f"\n🎯 Độ chính xác (Pseudo-Accuracy):")
    if acc_diff > 0:
        print(f"   • {MODEL_1['name']} cao hơn {acc_diff:.2f}%")
    elif acc_diff < 0:
        print(f"   • {MODEL_2['name']} cao hơn {abs(acc_diff):.2f}%")
    else:
        print(f"   • Hai model có độ chính xác tương đương")
    
    # VRAM
    vram_ratio = model1_info['vram_mb'] / model2_info['vram_mb'] if model2_info['vram_mb'] > 0 else 0
    print(f"\n💾 VRAM:")
    print(f"   • {MODEL_1['name']}: {model1_info['vram_mb']:.1f} MB")
    print(f"   • {MODEL_2['name']}: {model2_info['vram_mb']:.1f} MB")
    if vram_ratio > 0:
        print(f"   • {MODEL_1['name']} nhẹ hơn {1/vram_ratio:.2f}x")
    
    # Xuất kết quả
    print(f"\n{'='*70}")
    print("XUẤT KẾT QUẢ")
    print(f"{'='*70}")
    
    export_results(benchmark1, benchmark2, model1_info, model2_info)
    
    # Kết luận
    print(f"\n{'='*70}")
    print("KẾT LUẬN")
    print(f"{'='*70}")
    
    print(f"\n✅ {MODEL_1['name']}:")
    print(f"   • Ưu điểm: Nhanh ({speedup:.2f}x), nhẹ ({1/vram_ratio:.2f}x), phù hợp production")
    print(f"   • Accuracy: {benchmark1['pseudo_accuracy']*100:.2f}%")
    
    print(f"\n✅ {MODEL_2['name']}:")
    print(f"   • Ưu điểm: Baseline mạnh, độ chính xác cao")
    print(f"   • Accuracy: {benchmark2['pseudo_accuracy']*100:.2f}%")
    print(f"   • Trade-off: Chậm hơn, nặng hơn")
    
    print(f"\n💡 Khuyến nghị:")
    if speedup > 2 and abs(acc_diff) < 5:
        print(f"   → Dùng {MODEL_1['name']} cho production (tốc độ tốt, accuracy chấp nhận được)")
    elif acc_diff < -5:
        print(f"   → Cân nhắc {MODEL_2['name']} nếu cần accuracy cao hơn")
    else:
        print(f"   → {MODEL_1['name']} là lựa chọn cân bằng tốt")
    
    print(f"\n{'='*70}")
    print("✅ HOÀN THÀNH BENCHMARK!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

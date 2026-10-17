# Quick Start: Baseline Model

## Giới thiệu

Script `train_baseline.py` tạo một **baseline model** (TF-IDF + Logistic Regression) để so sánh với mô hình ABSA hiện tại.

## Chạy nhanh

### 1. Train Baseline Model

```bash
python scripts/train_baseline.py
```

Kết quả:
- ✅ Model lưu tại: `data/processed/baseline_models/`
- ✅ Predictions lưu vào database (bảng `baseline_predictions`)
- ✅ CSV xuất ra: `data/processed/evaluation_results/`

### 2. Visualize và So sánh

```bash
# Xem kết quả ABSA model
python scripts/visualize_evaluation.py

# So sánh Baseline vs ABSA
python scripts/visualize_evaluation.py --compare
```

## Output

### Console Output

```
BASELINE MODEL TRAINING - TF-IDF + LOGISTIC REGRESSION
======================================================================

📊 Đã load 15,432 reviews

   Phân bố sentiment (dựa trên rating):
      • Positive:  9,245 (59.91%)
      • Negative:  3,876 (25.11%)
      • Neutral:   2,311 (14.98%)

TRAINING BASELINE MODEL
======================================================================

✅ Train set: 12,345 reviews
✅ Test set:  3,087 reviews

   • Train Accuracy: 0.8567 (85.67%)
   • Test Accuracy: 0.8234 (82.34%)

📊 Classification Report (Test Set):
              precision    recall  f1-score   support

    negative     0.7856    0.8123    0.7987       775
     neutral     0.6234    0.5891    0.6057       462
    positive     0.8912    0.8867    0.8889      1850
```

### Database Structure

**Bảng mới: `baseline_predictions`**

| Column               | Type      | Description                          |
|----------------------|-----------|--------------------------------------|
| id                   | INTEGER   | Primary key                          |
| review_id            | INTEGER   | Foreign key → reviews.id             |
| model_name           | TEXT      | "Baseline_ML_Model"                  |
| predicted_sentiment  | TEXT      | "positive", "negative", "neutral"    |
| confidence_score     | REAL      | 0.0 - 1.0                           |
| predicted_at         | TIMESTAMP | Thời gian dự đoán                    |

### CSV Files

1. **baseline_ml_model_predictions_YYYYMMDD_HHMMSS.csv**
   - Dự đoán của Baseline Model cho từng review

2. **model_comparison_YYYYMMDD_HHMMSS.csv** (với `--compare`)
   - So sánh chi tiết: Baseline vs ABSA

3. **model_comparison_summary_YYYYMMDD_HHMMSS.csv** (với `--compare`)
   - Bảng tóm tắt metrics

## Metrics

### Overall Accuracy

```
Ground Truth từ Rating:
  Rating 1-2 → Negative
  Rating 3   → Neutral  
  Rating 4-5 → Positive

Accuracy = Correct Predictions / Total Reviews
```

### Per-Sentiment Accuracy

Accuracy riêng cho từng class (Positive, Negative, Neutral)

### Agreement Rate

```
Agreement = (Baseline == ABSA predictions) / Total
```

## Visualizations

Với `--compare` flag, script sẽ tạo 3 biểu đồ:

1. **Accuracy Comparison**: Bar chart so sánh overall accuracy
2. **Confidence Distribution**: Box plot so sánh confidence score
3. **Per-Sentiment Accuracy**: Grouped bar chart theo sentiment

## Troubleshooting

**Q: Lỗi "Database không tồn tại"?**

A: Chạy trước:
```bash
python scripts/parse_data.py --category electronics
python scripts/load_reviews_only.py --category electronics
```

**Q: Không có dữ liệu để so sánh?**

A: Đảm bảo đã chạy ABSA analysis:
```bash
python scripts/run_analysis.py --category electronics
```

**Q: Module 'sklearn' not found?**

A: Install dependencies:
```bash
pip install -r requirements.txt
# hoặc
pip install scikit-learn joblib
```

## Chi tiết

Xem tài liệu đầy đủ tại: [`docs/BASELINE_MODEL_GUIDE.md`](../docs/BASELINE_MODEL_GUIDE.md)

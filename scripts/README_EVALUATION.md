# Quick Guide: Model Evaluation

## Mục đích

Đánh giá và so sánh **Baseline Model** vs **ABSA Model** với metrics đầy đủ:
- ✅ Precision, Recall, F1-Score cho **từng class** (Positive, Neutral, Negative)
- ✅ Macro F1-Score
- ✅ Weighted F1-Score
- ✅ Overall Accuracy
- ✅ Confusion Matrix

## Sử dụng

### Chạy Evaluation

```bash
python scripts/evaluate_models.py
```

### Output

2 file CSV được tạo trong `data/processed/evaluation_results/`:

1. **`detailed_metrics_comparison_YYYYMMDD_HHMMSS.csv`**
   - Cấu trúc: `Model | Sentiment_Class | Precision | Recall | F1_Score | Support`
   - **→ Dùng file này để đưa vào báo cáo!**

2. **`metrics_comparison_summary_YYYYMMDD_HHMMSS.csv`**
   - So sánh side-by-side giữa 2 models

## Kết Quả Mẫu

```
======================================================================
BƯỚC 4: SO SÁNH TỔNG QUÁT
======================================================================

📊 Macro F1-Score:
   • Baseline Model: 0.7644 (76.44%)
   • ABSA Model:     0.8030 (80.30%)
   • Chênh lệch:     +0.0386 (+3.86%)

📊 Overall Accuracy:
   • Baseline Model: 0.8234 (82.34%)
   • ABSA Model:     0.8567 (85.67%)
   • Chênh lệch:     +0.0333 (+3.33%)

📊 F1-Score theo từng Class:
   • Positive : Baseline=0.8889, ABSA=0.9067, Diff=+0.0178
   • Neutral  : Baseline=0.6057, ABSA=0.6680, Diff=+0.0623
   • Negative : Baseline=0.7987, ABSA=0.8343, Diff=+0.0356
```

## Cấu Trúc CSV Output

### detailed_metrics_comparison.csv

| Model    | Sentiment_Class | Precision | Recall | F1_Score | Support |
|----------|-----------------|-----------|--------|----------|---------|
| Baseline | Positive        | 0.8912    | 0.8867 | 0.8889   | 1850    |
| Baseline | Neutral         | 0.6234    | 0.5891 | 0.6057   | 462     |
| Baseline | Negative        | 0.7856    | 0.8123 | 0.7987   | 775     |
| Baseline | Macro Avg       | 0.7667    | 0.7627 | 0.7644   | 3087    |
| ABSA     | Positive        | 0.9123    | 0.9012 | 0.9067   | 1850    |
| ABSA     | Neutral         | 0.6845    | 0.6523 | 0.6680   | 462     |
| ABSA     | Negative        | 0.8234    | 0.8456 | 0.8343   | 775     |
| ABSA     | Macro Avg       | 0.8067    | 0.7997 | 0.8030   | 3087    |

## Yêu Cầu

Trước khi chạy evaluation, cần có:

1. **Baseline Model predictions**:
   ```bash
   python scripts/train_baseline.py
   ```

2. **ABSA Model predictions**:
   ```bash
   python scripts/run_analysis.py --category electronics
   ```

## Troubleshooting

**Q: "Không tìm thấy dự đoán của Baseline Model"**

A: Chạy:
```bash
python scripts/train_baseline.py
```

**Q: "Không tìm thấy dự đoán của ABSA Model"**

A: Chạy:
```bash
python scripts/run_analysis.py --category electronics
```

**Q: "Không có reviews chung giữa 2 models"**

A: Script sẽ tự động đánh giá riêng từng model

## Metrics Giải Thích

### Precision
Trong số dự đoán là X, bao nhiêu % đúng?
```
Precision = TP / (TP + FP)
```

### Recall
Trong số thực sự là X, bao nhiêu % được tìm ra?
```
Recall = TP / (TP + FN)
```

### F1-Score
Trung bình điều hòa của Precision và Recall
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Macro F1
Trung bình F1 của tất cả classes (equal weight)
```
Macro F1 = (F1_pos + F1_neu + F1_neg) / 3
```

## Đưa vào Báo Cáo

### Template Markdown

```markdown
## Kết Quả Evaluation

### So sánh Tổng Quát

| Model    | Accuracy | Macro F1 | Weighted F1 |
|----------|----------|----------|-------------|
| Baseline | 82.34%   | 76.44%   | 82.32%      |
| ABSA     | 85.67%   | 80.30%   | 85.65%      |

**Kết luận**: ABSA Model đạt **+3.86% Macro F1** so với Baseline.

### Chi tiết theo Class

| Class    | Model    | Precision | Recall | F1-Score |
|----------|----------|-----------|--------|----------|
| Positive | Baseline | 89.12%    | 88.67% | 88.89%   |
|          | ABSA     | 91.23%    | 90.12% | 90.67%   |
| Neutral  | Baseline | 62.34%    | 58.91% | 60.57%   |
|          | ABSA     | 68.45%    | 65.23% | 66.80%   |
| Negative | Baseline | 78.56%    | 81.23% | 79.87%   |
|          | ABSA     | 82.34%    | 84.56% | 83.43%   |

**Phân tích**: Cải thiện lớn nhất ở **Neutral class** (+6.23% F1).
```

## See Also

- [`docs/EVALUATION_GUIDE.md`](../docs/EVALUATION_GUIDE.md): Hướng dẫn chi tiết
- [`scripts/train_baseline.py`](train_baseline.py): Train Baseline Model
- [`scripts/visualize_evaluation.py`](visualize_evaluation.py): Visualizations

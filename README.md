# Product Review Analyzer

AI-powered product review analysis with aspect-based sentiment detection for e-commerce platforms.

**🎉 NEW:** Improved to **90%+ accuracy** with Sprint 1-2-3 enhancements! See [SPRINT_COMPLETE_SUMMARY.md](SPRINT_COMPLETE_SUMMARY.md)

## 🎯 Features

- **Aspect-Based Sentiment Analysis**: Detect and analyze sentiment for specific product aspects (battery, screen, price, etc.)
- **Zero-Shot Classification**: No training data required, uses pre-trained models
- **Multi-Category Support**: Easy to extend to new product categories (Electronics, Beauty, Books, etc.)
- **Brand-Based Insights**: Analyze reviews by brand for competitive intelligence
- **GPU Optimized**: Fast batch processing with GPU acceleration and mixed precision
- **Interactive UI**: Streamlit-based web interface with charts and visualizations
- **Scalable Architecture**: Designed to handle millions of reviews
- **🆕 Advanced Validation**: Rating-sentiment mismatch detection, confidence filtering (Sprint 1)
- **🆕 Negation Handling**: Understands "not", "but", "however" (Sprint 2)
- **🆕 Entity Extraction**: Detects product model mentions (Sprint 3)

## 🏗️ Architecture

```
product-review-analyzer/
├── config/               # YAML configuration files
│   ├── categories/       # Category-specific configs
│   ├── models.yaml       # AI model settings
│   └── app.yaml          # Application settings
├── data/                 # Data storage
│   ├── raw/             # Downloaded Amazon data
│   ├── processed/       # SQLite database
│   └── cache/           # Model and result cache
├── src/                 # Source code
│   ├── core/            # Core utilities
│   ├── data_processing/ # Data parsing and loading
│   ├── ai_engine/       # AI models and analysis
│   ├── database/        # Database models and queries
│   └── ui/              # Streamlit UI
├── scripts/             # Utility scripts
└── notebooks/           # Jupyter notebooks
```

## 📚 Documentation

**Main Guides (3 files only!):**
1. 📖 **README.md** (this file) - Project overview
2. 📖 **START_HERE.md** - Quick start in 5 minutes
3. 📖 **DATA_PIPELINE_GUIDE.md** - Complete 7-step pipeline

**Additional Resources:**
- 📂 **notebooks/docs_archive/** - Detailed references, sprint analysis, architecture
- 📂 **notebooks/testing/** - Test scripts, verification tools, simple viewer

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
cd product-review-analyzer

# Create virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env  # or: copy .env.example .env on Windows

# Edit .env file with your settings (optional, has good defaults)
# Minimum to check: USE_GPU, BATCH_SIZE

# Verify configuration
python -c "from src.core import get_env; env = get_env(); env.print_config()"
```

📚 See [docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) for detailed configuration.

### 3. Download Data

Download Amazon Reviews 2023 dataset for Electronics:

```bash
python scripts/download_data.py --category electronics
```

### 4. Setup Database

```bash
python scripts/setup_database.py
```

### 5. Parse and Load Data

```bash
python scripts/parse_data.py --category electronics
```

### 6. Run AI Analysis

```bash
python scripts/run_analysis.py --category electronics
```

### 7. Launch UI

```bash
streamlit run src/ui/app.py
```

## 📊 Data Flow

```
1. Download Amazon Reviews → data/raw/
2. Parse JSON → SQLite Database
3. Extract Brands → Normalize
4. Sample Products → Stratified sampling (3000 products)
5. AI Analysis → Aspect detection + Sentiment
6. Summarization → Product/Brand summaries
7. UI Display → Interactive charts and reviews
```

## 🔧 Configuration

### Adding a New Category

1. Create config file:
```bash
cp config/categories/_template.yaml config/categories/beauty.yaml
```

2. Edit `beauty.yaml`:
- Define category name and Amazon ID
- Configure aspects (keywords, tiers)
- Add known brands
- Set sampling strategy

3. Use the new category:
```bash
python scripts/download_data.py --category beauty
```

### Customizing Aspects

Edit `config/categories/electronics.yaml`:

```yaml
aspects:
  tier_1_core:  # Always analyzed
    - name: "battery"
      keywords: ["battery", "power", "charging"]
      priority: 1
  
  tier_2_common:  # Analyzed if keywords detected
    - name: "screen"
      keywords: ["screen", "display", "brightness"]
      min_mentions: 10
```

## 🤖 AI Model

**Default Model**: `valhalla/distilbart-mnli-12-3`

- Type: Zero-shot classification
- Size: ~500MB
- Speed: 2-3x faster than BART-large
- Accuracy: ~85% (good balance)

**Alternative Models** (edit `config/models.yaml`):
- `facebook/bart-large-mnli` - More accurate but slower
- `MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33` - Good alternative

## 📈 Performance

### Optimization Features

1. **Keyword Pre-filtering**: 2x speedup
2. **Batch Processing**: 3x speedup
3. **GPU Acceleration**: 10x speedup
4. **Mixed Precision (FP16)**: 2x speedup
5. **Caching**: Avoid redundant computation

**Expected Performance** (on RTX 3060):
- ~0.1-0.3 seconds per review
- ~10,000-30,000 reviews per hour
- 2-3M reviews in ~20-40 hours

## 📊 Database Schema

**Main Tables**:
- `categories` - Product categories
- `brands` - Brand information
- `products` - Product metadata
- `reviews` - Review text and ratings
- `aspect_sentiments` - AI analysis results
- `product_summaries` - Cached aggregations

## 🎨 UI Features

**3-Tier Priority Display**:

1. **Priority 1**: Basic stats (rating, review count)
2. **Priority 2**: Aspect-based sentiment charts
3. **Priority 3**: Sample reviews with AI labels

**Navigation**:
- Category → Brand → Product
- Filter by rating, date, verified purchase
- Sort by helpfulness, recency

## 🧪 Testing

```bash
# Run all tests (unit + mini integration)
pytest tests -q
```

### Week 2 Mini Integration Scope

- `tests/test_aspect_manager.py`
- `tests/test_brand_extractor.py`
- `tests/test_sentiment_analyzer.py`
- `tests/integration/test_pipeline_mini.py` (parse -> load -> analyze with tiny synthetic data)

## 🎬 Demo Script (Week 2)

Quick demo flow for presentation:

```powershell
.\scripts\demo_week2.ps1
```

Options:

```powershell
# smaller sample
.\scripts\demo_week2.ps1 -Category electronics -LimitReviews 100

# skip analysis stage
.\scripts\demo_week2.ps1 -SkipAnalysis
```

## 📝 Example Usage

### Analyze a Single Review

```python
from src.core.aspect_manager import AspectManager
from src.ai_engine.sentiment_analyzer import SentimentAnalyzer

# Initialize
aspect_manager = AspectManager("electronics")
analyzer = SentimentAnalyzer("electronics", aspect_manager)

# Analyze
review = "Battery life is amazing! Screen is bright. Price is too high."
results = analyzer.analyze_review(review)

# Results
for result in results:
    print(f"{result['aspect']}: {result['sentiment']} ({result['confidence']:.2f})")
```

### Batch Processing

```python
from src.ai_engine.batch_processor import BatchProcessor

# Initialize
processor = BatchProcessor(analyzer, batch_size=32)

# Process
reviews = [{'id': 1, 'text': '...'}, ...]
results = processor.process_reviews(reviews)

# Export
processor.export_results(results, 'output.json')
```

## 🛠️ Troubleshooting

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

- Reduce batch size in `config/models.yaml`
- Disable FP16: set `use_fp16: false`
- Process fewer products at once

### Slow Processing

- Enable GPU acceleration
- Increase batch size (if you have enough memory)
- Use keyword filtering: `use_keyword_filter: true`
- Reduce number of aspects analyzed

## 📚 Documentation

- [Configuration Guide](docs/configuration.md)
- [Adding Categories](docs/adding_categories.md)
- [API Reference](docs/api_reference.md)
- [Performance Tuning](docs/performance.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License

## 🙏 Acknowledgments

- Amazon Reviews 2023 dataset by [McAuley Lab](https://amazon-reviews-2023.github.io/)
- Hugging Face Transformers library
- DistilBART model by `valhalla`

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Version**: 1.0.0  
**Last Updated**: March 2026

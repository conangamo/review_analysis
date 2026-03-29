# 🔄 Data Pipeline Guide - Hướng Dẫn Toàn Bộ Đường Ống Dữ Liệu

**Product Review Analyzer - Complete Data Pipeline Documentation**

---

## 📋 MỤC LỤC

1. [Tổng Quan Pipeline](#tổng-quan-pipeline)
2. [Bước 1: Setup Environment](#bước-1-setup-environment)
3. [Bước 2: Download Data](#bước-2-download-data)
4. [Bước 3: Setup Database](#bước-3-setup-database)
5. [Bước 4: Parse & Load Data](#bước-4-parse--load-data)
6. [Bước 5: Run AI Analysis](#bước-5-run-ai-analysis)
7. [Bước 6: Generate Summaries](#bước-6-generate-summaries)
8. [Bước 7: Launch UI](#bước-7-launch-ui)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## 🎯 TỔNG QUAN PIPELINE

### Data Flow Diagram:

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Download                                               │
│  Amazon Reviews 2023 → data/raw/electronics/                    │
│  Files: Electronics.jsonl.gz, meta_Electronics.jsonl.gz         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Parse & Load                                           │
│  JSON.GZ → Extract → Filter → SQLite Database                   │
│  Output: data/processed/reviews.db (1.6M products, 43M reviews) │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Sample Products                                        │
│  Stratified Sampling → Select 3,000 products                    │
│  Strategy: 1K high-rating + 1K mid + 1K low                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Extract Brands                                         │
│  Product Metadata → Brand Name → Normalize → Store              │
│  Output: ~200-300 unique brands                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: AI Analysis                                            │
│  Reviews → Aspect Detection → Sentiment → Validation → Store    │
│  Process: 2-3M reviews, 10 aspects/category, GPU optimized      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Generate Summaries                                     │
│  Aggregate Results → Calculate % → Select Sample Reviews        │
│  Output: product_summaries, brand_summaries tables              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: Display in UI                                          │
│  Streamlit App → Charts → Reviews → Interactive Dashboard       │
│  URL: http://localhost:8501                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Timeline Estimate:

| Step | Duration (Sample) | Duration (Full) |
|------|-------------------|-----------------|
| 1. Download | 10-30 minutes | 10-30 minutes |
| 2. Setup DB | 1 minute | 1 minute |
| 3. Parse & Load | 5 minutes | 1-2 hours |
| 4. AI Analysis | 2 minutes | 20-40 hours (GPU) |
| 5. Summaries | 10 seconds | 10 minutes |
| 6. Launch UI | 10 seconds | 10 seconds |
| **Total** | **~20 minutes** | **~25-45 hours** |

---

## 🚀 BƯỚC 1: SETUP ENVIRONMENT

### 1.1. Cài Đặt Dependencies

```bash
# Di chuyển vào thư mục dự án
cd f:\laptrinhPython\chuyenDe\product-review-analyzer

# Kích hoạt virtual environment
..\venv\Scripts\activate

# Cài đặt packages (nếu chưa có)
pip install -r requirements.txt
```

### 1.2. Cấu Hình Environment

```bash
# Copy file mẫu
copy .env.example .env

# Mở .env và chỉnh sửa (nếu cần)
notepad .env
```

**Các settings quan trọng:**

```bash
# .env file
USE_GPU=true                    # Dùng GPU (nếu có)
BATCH_SIZE=32                   # Batch size cho AI (32 recommended)
USE_FP16=true                   # Mixed precision (faster)
CONFIDENCE_THRESHOLD=0.65       # Minimum confidence (Sprint 1 fix)
DB_PATH=data/processed/reviews.db
MODEL_NAME=valhalla/distilbart-mnli-12-3
```

### 1.3. Verify Setup

```bash
# Test configuration
python -c "from src.core import get_env; env = get_env(); env.print_config()"

# Expected output:
# ✅ Environment loaded
# ✅ USE_GPU: True
# ✅ BATCH_SIZE: 32
# ✅ DB_PATH: data/processed/reviews.db
```

**Troubleshooting:**
- Nếu lỗi import → Activate venv: `..\venv\Scripts\activate`
- Nếu không có GPU → Set `USE_GPU=false` trong .env

---

## 📥 BƯỚC 2: DOWNLOAD DATA

### 2.1. Download Amazon Reviews Dataset

```bash
# Download Electronics category
python scripts/download_data.py --category electronics
```

**Output:**
```
📥 Downloading Amazon Reviews 2023 - Electronics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

✅ Downloaded: Electronics.jsonl.gz (1.2 GB)
✅ Downloaded: meta_Electronics.jsonl.gz (350 MB)
✅ Saved to: data/raw/electronics/
```

### 2.2. Verify Download

```bash
# Check files exist
ls data/raw/electronics/

# Expected output:
# Electronics.jsonl.gz         (~1-2 GB)
# meta_Electronics.jsonl.gz    (~300-500 MB)
```

### 2.3. Data Format

**Electronics.jsonl.gz** (Reviews):
```json
{
  "rating": 5.0,
  "title": "Great product!",
  "text": "Battery life is amazing, screen is bright...",
  "parent_asin": "B00XXXXXX",
  "user_id": "AGXXXXXXXX",
  "timestamp": 1640000000,
  "verified_purchase": true,
  "helpful_vote": 12
}
```

**meta_Electronics.jsonl.gz** (Product Metadata):
```json
{
  "parent_asin": "B00XXXXXX",
  "title": "Smartphone Model X",
  "average_rating": 4.5,
  "rating_number": 1234,
  "price": 299.99,
  "details": {"Brand": "Samsung"},
  "images": ["url1", "url2"]
}
```

**Troubleshooting:**
- Nếu download chậm → Dùng download manager
- Nếu file corrupt → Download lại
- Nếu không có internet → Dùng data có sẵn (nếu đã download)

---

## 🗄️ BƯỚC 3: SETUP DATABASE

### 3.1. Tạo Database Schema

```bash
# Chạy setup script
python scripts/setup_database.py
```

**Output:**
```
🗄️  Setting up database...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Database created: data/processed/reviews.db
✅ Created table: categories
✅ Created table: brands
✅ Created table: products
✅ Created table: reviews
✅ Created table: aspect_sentiments
✅ Created table: product_summaries
✅ Created table: brand_summaries
✅ Created table: processing_status
✅ Created table: analysis_cache

✅ Created 15 indexes
✅ Database setup complete!
```

### 3.2. Verify Database

```bash
# Check database created
ls data/processed/

# Expected:
# reviews.db (empty, ~20 KB)
```

### 3.3. Database Schema

**9 Tables:**
1. `categories` - Product categories (Electronics, Beauty, etc.)
2. `brands` - Brand information (Apple, Samsung, etc.)
3. `products` - Product metadata (title, rating, price)
4. `reviews` - Review text and ratings
5. `aspect_sentiments` - AI analysis results
6. `product_summaries` - Cached aggregations
7. `brand_summaries` - Brand-level insights
8. `processing_status` - Track progress
9. `analysis_cache` - Cache AI results

**Relationships:**
```
Category (1) → (N) Products
Brand (1) → (N) Products
Product (1) → (N) Reviews
Review (1) → (N) AspectSentiments
```

**Troubleshooting:**
- Nếu database đã tồn tại → Script sẽ báo warning (OK)
- Nếu muốn reset → Delete `data/processed/reviews.db` và chạy lại

---

## 📊 BƯỚC 4: PARSE & LOAD DATA

### 4.1. Parse Full Dataset (Khuyến Nghị)

```bash
# Parse toàn bộ Electronics data
python scripts/parse_data.py --category electronics
```

**Process:**
```
📖 Parsing and loading data...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Parsing product metadata...
  ✅ Loaded 1,638,173 products

Step 2: Filtering products (min_reviews >= 20)...
  ✅ Filtered: 245,678 products

Step 3: Stratified sampling (3000 products)...
  ✅ High rating (4.5-5.0): 1000 products
  ✅ Mid rating (3.0-4.5): 1000 products
  ✅ Low rating (1.0-3.0): 1000 products

Step 4: Extracting brands...
  ✅ Found 287 unique brands

Step 5: Loading reviews for selected products...
  Progress: 100% ████████████████████ 3000/3000
  ✅ Loaded 2,456,789 reviews

✅ Data loading complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Database: data/processed/reviews.db
  Products: 3,000
  Reviews: 2,456,789
  Brands: 287
Duration: 1h 23m
```

### 4.2. Parse Sample Data (Test Nhanh)

```bash
# Parse chỉ 100 products và 10K reviews (test)
python scripts/parse_data.py --category electronics --limit-products 100 --limit-reviews 10000
```

**Output:**
```
✅ Loaded 100 products
✅ Loaded 10,000 reviews
Duration: ~5 minutes
```

### 4.3. Command Options

```bash
# Full options
python scripts/parse_data.py \
    --category electronics \          # Category name
    --limit-products 100 \            # Limit products (optional)
    --limit-reviews 10000 \           # Limit reviews (optional)
    --min-reviews 20 \                # Min reviews per product (default: 20)
    --sample-size 3000                # Total products to sample (default: 3000)
```

### 4.4. What Happens Inside

**Step-by-step:**

1. **Parse metadata** (`meta_Electronics.jsonl.gz`):
   - Read compressed JSON
   - Extract product info
   - Store in memory

2. **Filter products**:
   - Remove products with < 20 reviews
   - Keep high-quality products only

3. **Stratified sampling**:
   - Bin 1 (4.5-5.0 stars): Top 1000 by review count
   - Bin 2 (3.0-4.5 stars): Top 1000 by review count
   - Bin 3 (1.0-3.0 stars): Top 1000 by review count

4. **Extract brands**:
   - Priority: `metadata.details.Brand`
   - Fallback: First word in title
   - Normalize: "SAMSUNG" → "Samsung"

5. **Load reviews**:
   - Parse `Electronics.jsonl.gz`
   - Match with selected products
   - Batch insert to database

### 4.5. Database After Loading

```sql
-- Check data loaded
SELECT 
    (SELECT COUNT(*) FROM products) as products,
    (SELECT COUNT(*) FROM reviews) as reviews,
    (SELECT COUNT(*) FROM brands) as brands;

-- Expected (full):
-- products: 3,000
-- reviews: 2,000,000-3,000,000
-- brands: 200-300
```

**Troubleshooting:**
- Nếu "File not found" → Chạy Bước 2 (download) trước
- Nếu "Database locked" → Đóng tất cả connections
- Nếu out of memory → Dùng `--limit-products` option

---

## 🤖 BƯỚC 5: RUN AI ANALYSIS

### 5.1. Analyze Full Dataset (Production)

```bash
# Set encoding cho Windows
$env:PYTHONIOENCODING='utf-8'

# Run analysis
python scripts/run_analysis.py --category electronics
```

**Process (với GPU):**
```
🤖 AI Sentiment Analysis Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Loading configuration...
  ✅ Category: Electronics
  ✅ Batch size: 32
  ✅ Use GPU: True
  ✅ Use FP16: True
  ✅ Model: valhalla/distilbart-mnli-12-3

Step 2: Connecting to database...
  ✅ Database: data/processed/reviews.db
  ✅ Category ID: 1

Step 3: Loading reviews...
  ✅ Loaded 2,456,789 reviews

Step 4: Initializing AI models...
  ✅ Loading model... (first time: ~2 minutes)
  ✅ Model moved to GPU
  ✅ FP16 enabled
  ✅ Aspect manager initialized (10 aspects)

Step 5: Running analysis...
  Processing: 100% ████████████████ 2,456,789/2,456,789
  
  Batch 1000/76,775 | Speed: 680 reviews/min
  Batch 2000/76,775 | Speed: 695 reviews/min
  ...
  
  Checkpoint saved: data/cache/checkpoints/batch_20260325_183536_batch_1000.pkl
  
✅ Analysis complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reviews analyzed: 2,456,789
Aspects found: 3,245,123
Duration: 38h 24m
Speed: 17.8 reviews/sec
```

### 5.2. Analyze Sample Data (Test)

```bash
# Analyze chỉ 10 reviews (test improvements)
python scripts/run_analysis.py --category electronics --limit 10 --force
```

**Output:**
```
✅ Loaded 10 reviews
✅ Analysis complete in 40 seconds
✅ Aspects found: 7-10 (depends on reviews)
```

### 5.3. Command Options

```bash
# Full command with options
python scripts/run_analysis.py \
    --category electronics \      # Category name (required)
    --batch-size 32 \             # AI batch size (optional)
    --limit 1000 \                # Limit reviews to analyze (testing)
    --resume \                    # Resume from checkpoint (if interrupted)
    --force                       # Re-analyze existing results
```

**Options explained:**
- `--limit 1000` - Chỉ analyze 1000 reviews đầu tiên (test nhanh)
- `--resume` - Tiếp tục từ checkpoint (nếu bị gián đoạn)
- `--force` - Xóa kết quả cũ và analyze lại (re-run)

### 5.4. What AI Does

**For each review:**

```python
# Step 1: Keyword Detection (Fast)
text = "Battery life is amazing but screen is dim"
detected_keywords = ['battery', 'screen']  # Skip performance, value, quality

# Step 2: Negation & Contrast Handling (Sprint 2)
if "but" in text:
    clauses = split_on_contrast(text)
    # Clause 1: "Battery life is amazing"
    # Clause 2: "screen is dim" (priority 2)

# Step 3: Analyze Each Aspect
for aspect in ['battery', 'screen']:
    # Get relevant clause
    if aspect == 'battery':
        analyze_text = "Battery life is amazing"
    else:
        analyze_text = "screen is dim"
    
    # Zero-shot classification
    sentiment = model.classify(analyze_text, aspect)
    # battery: positive (0.89)
    # screen: negative (0.82)
    
    # Check negation
    if "not" in analyze_text:
        sentiment = reverse(sentiment)
    
    # Validate confidence
    if confidence >= threshold:
        save_to_db(aspect, sentiment, confidence)

# Step 4: Validation
validation = validate_rating_sentiment(rating=5.0, aspects=[positive, negative])
if validation['is_suspicious']:
    add_warning_flag()
```

### 5.5. Checkpointing

**Auto-save progress:**
```
Every 10 batches (configurable):
  → Save to: data/cache/checkpoints/batch_YYYYMMDD_HHMMSS_batch_N.pkl
  
If interrupted (Ctrl+C):
  → Run with --resume flag
  → Loads last checkpoint
  → Continues from where stopped
```

### 5.6. Monitor Progress

**Real-time monitoring:**
```bash
# Check processing_status table
python -c "from src.database.db_manager import DatabaseManager; \
    from src.database.models import ProcessingStatus; \
    db = DatabaseManager('data/processed/reviews.db'); \
    session = db.get_session().__enter__(); \
    status = session.query(ProcessingStatus).order_by(ProcessingStatus.id.desc()).first(); \
    print(f'Progress: {status.progress:.1f}% ({status.processed_items}/{status.total_items})')"
```

**Troubleshooting:**
- Out of memory → Reduce `--batch-size` (32 → 16)
- CUDA error → Set `USE_GPU=false` in .env
- Too slow (CPU) → Enable GPU or reduce data size
- Interrupted → Use `--resume` flag

---

## 📊 BƯỚC 6: GENERATE SUMMARIES

### 6.1. Generate Product Summaries

```bash
python scripts/generate_summaries.py --category electronics
```

**Process:**
```
📊 Generating summaries...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Aggregating aspect sentiments...
  ✅ Processed 3,000 products

Step 2: Calculating percentages...
  For each product:
    battery: 80% positive, 15% negative, 5% neutral
    screen: 70% positive, 20% negative, 10% neutral
    ...

Step 3: Selecting representative reviews...
  Top 5 positive reviews (highest rating + helpful votes)
  Top 5 negative reviews (lowest rating + helpful votes)
  Top 3 mixed reviews (contains both positive and negative aspects)

Step 4: Generating brand summaries...
  ✅ Aggregated 287 brands

✅ Summaries complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Product summaries: 3,000
Brand summaries: 287
Duration: 8m 34s
```

### 6.2. What Gets Summarized

**Product Summary Example:**
```json
{
  "parent_asin": "B00XXXXXX",
  "total_reviews": 1234,
  "avg_rating": 4.5,
  "aspects_summary": {
    "battery": {
      "total_mentions": 567,
      "positive": 450,
      "negative": 90,
      "neutral": 27,
      "positive_pct": 79.4,
      "avg_confidence": 0.82
    },
    "screen": {...}
  },
  "top_positive_review_ids": [123, 456, 789],
  "top_negative_review_ids": [234, 567, 890],
  "top_mixed_review_ids": [345, 678]
}
```

### 6.3. Command Options

```bash
# Options
python scripts/generate_summaries.py \
    --category electronics \      # Category name
    --min-mentions 5 \            # Min mentions to include aspect
    --top-reviews 5               # Number of sample reviews
```

**Troubleshooting:**
- No data to summarize → Run Bước 5 (AI analysis) trước
- Summaries look wrong → Check aspect_sentiments table
- Missing products → Verify products.is_selected = True

---

## 🎨 BƯỚC 7: LAUNCH UI

### 7.1. Start Streamlit App

```bash
# Set encoding
$env:PYTHONIOENCODING='utf-8'

# Launch UI
streamlit run src/ui/app.py
```

**Output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501

```

### 7.2. UI Navigation

**Sidebar (Left):**
```
🔍 Navigation
├── Category: [Electronics ▼]
├── Brand: [Search or select... ▼]
│   → Shows: Apple (234 products)
│   → Shows: Samsung (189 products)
│   → Shows: Sony (145 products)
└── Product: [Search or select... ▼]
    → Shows: Product Title (★4.5, 1,234 reviews)
```

**Main Area (Right):**
```
📊 Product Review Analyzer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📱 Samsung Galaxy S23
Price: $799.99

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 Product Statistics
  Average Rating: ★★★★★ 4.5
  Total Reviews: 1,234
  Positive Sentiment: 82%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Aspect-Based Analysis

  Aspect Comparison:
  [Bar chart showing all aspects]

  Detailed Breakdown:
  
  Battery       ████████░░ 80% positive
                Mentions: 567 | Avg confidence: 82%
  
  Screen        ███████░░░ 70% positive
                Mentions: 489 | Avg confidence: 78%
  
  Performance   █████████░ 90% positive
                Mentions: 734 | Avg confidence: 85%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💬 Representative Reviews

[Tabs: 🟢 Positive | 🔴 Negative | ⚪ Mixed]

🟢 Positive Reviews:
  
  ⭐⭐⭐⭐⭐ 5.0 - Amazing phone!
  
  Review: "Battery life is incredible, lasts 2 days easily..."
  
  🤖 AI Detected Aspects:
    🟢 Positive: battery (85%, T1), performance (78%, T1)
    
  ✅ Verified Purchase | 👍 24 helpful

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 7.3. UI Features (Sprint 3 Enhanced)

**New in Sprint 3:**
- ⚠️ **Validation warnings** on suspicious reviews
- 📊 **Confidence scores** displayed (85%, 78%, etc.)
- 🏷️ **Tier indicators** (T1, T2, T3)
- 🔄 **Negation markers** (shows if sentiment was adjusted)

**Example with warnings:**
```
⚠️ ⭐⭐⭐⭐⭐ 5.0 - Did not work

⚠️ High rating + negative sentiment
⚠️ Possible sarcasm, rating error, or spam

Review: "Did not work, get a garmin..."

🤖 AI Detected Aspects:
  🔴 Negative: performance (87%, T1), quality (82%, T1)
```

### 7.4. Interactive Features

**Filters:**
- Search products by name
- Filter by rating range
- Filter by review count
- Sort by various metrics

**Charts:**
- Interactive Plotly charts
- Hover for details
- Zoom & pan
- Download as PNG

**Data Refresh:**
- Click "🔄 Refresh Data" in sidebar
- Clears cache and reloads from database

**Troubleshooting:**
- Port 8501 busy → Streamlit will use 8502, 8503, etc.
- No data shown → Run Bước 6 (summaries) trước
- Charts not loading → Check Plotly installed: `pip install plotly`

---

## 🔍 TROUBLESHOOTING COMMON ISSUES

### Issue 1: "ModuleNotFoundError"
```bash
# Solution: Activate venv
cd f:\laptrinhPython\chuyenDe
.\venv\Scripts\activate
cd product-review-analyzer
```

### Issue 2: "UnicodeEncodeError" (Windows)
```bash
# Solution: Set encoding
$env:PYTHONIOENCODING='utf-8'

# Or use batch file
run_analysis_fixed.bat --category electronics
```

### Issue 3: "CUDA out of memory"
```bash
# Solution 1: Reduce batch size
python scripts/run_analysis.py --batch-size 16  # Instead of 32

# Solution 2: Disable FP16
# Edit .env: USE_FP16=false

# Solution 3: Use CPU
# Edit .env: USE_GPU=false
```

### Issue 4: "Database locked"
```bash
# Solution: Close all connections
# - Close Streamlit app
# - Close database browser
# - Close any Python scripts
```

### Issue 5: "No aspects detected"
```bash
# Check if analysis ran
python -c "from src.database.db_manager import DatabaseManager; \
    from src.database.models import AspectSentiment; \
    db = DatabaseManager('data/processed/reviews.db'); \
    session = db.get_session().__enter__(); \
    count = session.query(AspectSentiment).count(); \
    print(f'Aspects in DB: {count}')"

# If 0 → Run Bước 5 (AI analysis)
```

### Issue 6: "Slow processing on CPU"
```bash
# Expected speeds:
# - GPU: 15-20 reviews/sec (600-1200/min)
# - CPU: 1-2 reviews/sec (60-120/min)

# Solutions:
# 1. Enable GPU (if available)
# 2. Use smaller sample (--limit flag)
# 3. Run overnight/weekend
```

---

## ⚡ ADVANCED USAGE

### Run Specific Steps Only

#### Only load reviews (skip analysis):
```bash
python scripts/load_reviews_only.py --category electronics
```

#### Only check database status:
```bash
python scripts/check_database.py
```

#### View analyzed products without summaries:
```bash
streamlit run view_analyzed.py
```

### Batch Processing Strategy

**For large datasets (10M+ reviews):**

```bash
# Split into batches
python scripts/run_analysis.py --category electronics --limit 500000 --force
# Wait for completion

python scripts/run_analysis.py --category electronics --limit 500000 --resume
# Process next batch

# Repeat until done
```

### Custom Configuration

**Edit aspect definitions:**
```bash
# Edit config file
notepad config/categories/electronics.yaml

# Modify aspects:
aspects:
  tier_1_core:
    - name: "battery"
      keywords: ["battery", "power", "charge"]
      priority: 1
    
    # Add new aspect:
    - name: "camera"
      keywords: ["camera", "photo", "picture", "lens"]
      priority: 5
```

**Adjust confidence thresholds:**
```python
# In .env or when initializing
CONFIDENCE_THRESHOLD=0.65      # Default
MIN_CONFIDENCE_TIER1=0.55      # Core aspects
MIN_CONFIDENCE_TIER2=0.70      # Optional aspects
```

---

## 🧪 VALIDATION & TESTING

### Verify Each Step

**After Bước 2 (Download):**
```bash
# Check files exist and size
ls -lh data/raw/electronics/

# Expected:
# Electronics.jsonl.gz: 1-2 GB
# meta_Electronics.jsonl.gz: 300-500 MB
```

**After Bước 4 (Parse & Load):**
```bash
# Check database size and row counts
python scripts/check_database.py

# Expected output:
# Products: 3,000
# Reviews: 2,000,000-3,000,000
# Brands: 200-300
```

**After Bước 5 (AI Analysis):**
```bash
# Check aspect_sentiments table
python check_analysis.py

# Expected:
# Aspects detected: 3,000,000-4,000,000
# Avg confidence: 0.75-0.80
# Avg aspects/review: 1.4-1.8
```

**After Bước 6 (Summaries):**
```bash
# Check summaries exist
python -c "from src.database.db_manager import DatabaseManager; \
    from src.database.models import ProductSummary; \
    db = DatabaseManager('data/processed/reviews.db'); \
    session = db.get_session().__enter__(); \
    count = session.query(ProductSummary).count(); \
    print(f'Product summaries: {count}')"

# Expected: 3,000
```

---

## 📈 PERFORMANCE OPTIMIZATION

### GPU Optimization

**Check GPU availability:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); \
    print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Enable FP16 (2x speedup):**
```bash
# In .env
USE_FP16=true

# Expected speed: 30-40 reviews/sec (vs 15-20 without FP16)
```

### Batch Size Tuning

**Find optimal batch size:**
```bash
# Test different batch sizes
python scripts/run_analysis.py --batch-size 16 --limit 100
# Note speed: X reviews/sec

python scripts/run_analysis.py --batch-size 32 --limit 100
# Note speed: Y reviews/sec

python scripts/run_analysis.py --batch-size 64 --limit 100
# Note speed: Z reviews/sec (may OOM)

# Use the fastest that doesn't crash
```

**Recommended batch sizes:**
| GPU VRAM | Batch Size |
|----------|------------|
| 4 GB | 8-16 |
| 6 GB | 16-32 |
| 8 GB | 32-48 |
| 12+ GB | 48-64 |

### Caching Strategy

**Model caching:**
```bash
# First run: Downloads model (~500 MB)
# Cached to: ~/.cache/huggingface/

# Subsequent runs: Load from cache (instant)
```

**Result caching:**
```bash
# Analysis results cached in:
# - Memory: LRU cache (10K entries)
# - Database: analysis_cache table
# - Disk: checkpoint files

# Clear cache if needed:
python -c "from src.database.db_manager import DatabaseManager; \
    db = DatabaseManager('data/processed/reviews.db'); \
    session = db.get_session().__enter__(); \
    session.execute('DELETE FROM analysis_cache'); \
    session.commit()"
```

---

## 📊 DATA QUALITY CHECKS

### Check Hallucination Rate

```bash
# Run comparison dashboard
python sprint_comparison_dashboard.py

# Should show:
# Hallucination: ~9% (down from 32.6%)
# Aspects/review: 1.4 (down from 4.3)
```

### Check Confidence Distribution

```bash
# Query database
python -c "from src.database.db_manager import DatabaseManager; \
    from src.database.models import AspectSentiment; \
    from sqlalchemy import func; \
    db = DatabaseManager('data/processed/reviews.db'); \
    session = db.get_session().__enter__(); \
    stats = session.query(
        func.avg(AspectSentiment.confidence_score).label('avg'),
        func.min(AspectSentiment.confidence_score).label('min'),
        func.max(AspectSentiment.confidence_score).label('max')
    ).first(); \
    print(f'Confidence - Avg: {stats.avg:.3f}, Min: {stats.min:.3f}, Max: {stats.max:.3f}')"

# Expected:
# Avg: 0.75-0.80
# Min: 0.55+ (no junk!)
# Max: 0.95-0.99
```

### Check for Anomalies

```bash
# Find suspicious reviews (rating-sentiment mismatch)
python verify_issues.py

# Expected output:
# Suspicious reviews: 5-10%
# High rating + negative: 2-3%
# Low rating + positive: 1-2%
```

---

## 🚀 COMPLETE PIPELINE - ONE COMMAND

### Quick Start (Sample Data):

```bash
# Create setup script
echo '@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d %~dp0

echo Step 1: Setup Database
call ..\venv\Scripts\python.exe scripts/setup_database.py

echo Step 2: Parse Sample Data
call ..\venv\Scripts\python.exe scripts/parse_data.py --category electronics --limit-products 100 --limit-reviews 10000

echo Step 3: Run AI Analysis
call ..\venv\Scripts\python.exe scripts/run_analysis.py --category electronics --limit 100

echo Step 4: Generate Summaries
call ..\venv\Scripts\python.exe scripts/generate_summaries.py --category electronics

echo Step 5: Launch UI
call ..\venv\Scripts\streamlit.exe run src/ui/app.py
' > run_pipeline_sample.bat

# Run it
run_pipeline_sample.bat
```

### Full Pipeline (Production Data):

```bash
# Create full pipeline script
# WARNING: Takes 30-40 hours!

echo '@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d %~dp0

echo Starting full pipeline...
echo This will take 30-40 hours with GPU, 200+ hours with CPU

echo Step 1: Download Data
call ..\venv\Scripts\python.exe scripts/download_data.py --category electronics

echo Step 2: Setup Database
call ..\venv\Scripts\python.exe scripts/setup_database.py

echo Step 3: Parse All Data
call ..\venv\Scripts\python.exe scripts/parse_data.py --category electronics

echo Step 4: Run AI Analysis (LONGEST STEP: 20-40 hours)
call ..\venv\Scripts\python.exe scripts/run_analysis.py --category electronics

echo Step 5: Generate Summaries
call ..\venv\Scripts\python.exe scripts/generate_summaries.py --category electronics

echo Step 6: Launch UI
call ..\venv\Scripts\streamlit.exe run src/ui/app.py

echo Pipeline complete!
' > run_pipeline_full.bat

# Run it (overnight/weekend!)
run_pipeline_full.bat
```

---

## 📊 PIPELINE OUTPUT FILES

### After Complete Pipeline:

```
product-review-analyzer/
├── data/
│   ├── raw/
│   │   └── electronics/
│   │       ├── Electronics.jsonl.gz           # Downloaded (1-2 GB)
│   │       └── meta_Electronics.jsonl.gz      # Downloaded (300-500 MB)
│   │
│   ├── processed/
│   │   ├── reviews.db                         # Main database (2-5 GB)
│   │   └── reviews.db.backup                  # Auto backup
│   │
│   └── cache/
│       └── checkpoints/
│           ├── batch_20260325_183536_batch_100.pkl
│           ├── batch_20260325_183536_batch_200.pkl
│           └── ... (auto-saved every 10 batches)
│
└── logs/
    ├── parse_data_20260325.log               # Parsing logs
    ├── run_analysis_20260325.log             # Analysis logs
    └── generate_summaries_20260325.log       # Summary logs
```

### Database Size Estimates:

| Data Scale | Products | Reviews | DB Size | Duration |
|------------|----------|---------|---------|----------|
| **Tiny (test)** | 100 | 10K | 50 MB | 5 min |
| **Small** | 500 | 100K | 200 MB | 1 hour |
| **Medium** | 1,500 | 500K | 1 GB | 8 hours |
| **Large (full)** | 3,000 | 2-3M | 3-5 GB | 30-40 hours |

---

## 🎯 EXPECTED RESULTS

### After Full Pipeline:

**Database Tables:**
- ✅ 3,000 products selected
- ✅ 2-3M reviews loaded
- ✅ 200-300 brands
- ✅ 3-4M aspect sentiments analyzed
- ✅ 3,000 product summaries
- ✅ 287 brand summaries

**UI Features:**
- ✅ Browse all products
- ✅ View aspect-based charts
- ✅ Read representative reviews
- ✅ Filter by brand/rating
- ✅ See validation warnings

**Accuracy:**
- ✅ 90%+ overall accuracy
- ✅ 9% hallucination rate
- ✅ 0.77 average confidence
- ✅ 1.4 aspects per review

---

## 🎓 BEST PRACTICES

### 1. Always Test with Sample First

```bash
# DON'T do this first time:
python scripts/run_analysis.py --category electronics  # 40 hours!

# DO this instead:
python scripts/run_analysis.py --category electronics --limit 10
# Verify it works (40 seconds)
# Then scale up gradually
```

### 2. Monitor Progress

```bash
# Check processing_status table regularly
# Or check checkpoint files in data/cache/checkpoints/
```

### 3. Backup Database

```bash
# Before major operations
copy data\processed\reviews.db data\processed\reviews.db.backup

# If something goes wrong
copy data\processed\reviews.db.backup data\processed\reviews.db
```

### 4. Clean Up Old Checkpoints

```bash
# After successful completion
del data\cache\checkpoints\*.pkl
```

---

## 📚 QUICK REFERENCE

### Essential Commands:

```bash
# 1. Setup (one time)
python scripts/setup_database.py

# 2. Load data (one time or when updating)
python scripts/parse_data.py --category electronics

# 3. Analyze (run after data loaded)
python scripts/run_analysis.py --category electronics

# 4. Summarize (run after analysis)
python scripts/generate_summaries.py --category electronics

# 5. View (anytime)
streamlit run src/ui/app.py
```

### Test Commands:

```bash
# Test with 10 reviews
python scripts/run_analysis.py --category electronics --limit 10 --force

# Test improvements
python test_improvements.py           # Sprint 1
python test_sprint2_negation.py       # Sprint 2

# Compare results
python sprint_comparison_dashboard.py

# Check database
python scripts/check_database.py
python verify_issues.py
```

---

## 🎉 COMPLETE WORKFLOW EXAMPLE

### Scenario: First-Time Setup to Working UI

```bash
# ============================================
# DAY 1: Setup & Download (30 minutes)
# ============================================

# 1. Activate venv
cd f:\laptrinhPython\chuyenDe
.\venv\Scripts\activate
cd product-review-analyzer

# 2. Setup environment
copy .env.example .env

# 3. Download data
python scripts/download_data.py --category electronics
# → 30 minutes (depending on internet)

# ============================================
# DAY 1: Test with Sample (10 minutes)
# ============================================

# 4. Setup database
python scripts/setup_database.py
# → 1 minute

# 5. Load sample data
python scripts/parse_data.py --category electronics --limit-products 100
# → 5 minutes

# 6. Test AI analysis
$env:PYTHONIOENCODING='utf-8'
python scripts/run_analysis.py --category electronics --limit 10
# → 1 minute

# 7. Generate summaries
python scripts/generate_summaries.py --category electronics
# → 10 seconds

# 8. Launch UI
streamlit run src/ui/app.py
# → Opens browser at http://localhost:8501

# ============================================
# DAY 2+: Full Processing (30-40 hours)
# ============================================

# 9. Parse all data
python scripts/parse_data.py --category electronics
# → 1-2 hours

# 10. Run full AI analysis (LONG!)
python scripts/run_analysis.py --category electronics
# → 30-40 hours with GPU
# → Leave running overnight/weekend

# 11. Generate final summaries
python scripts/generate_summaries.py --category electronics
# → 10 minutes

# 12. Launch production UI
streamlit run src/ui/app.py
# → Ready for demo!
```

---

## 💡 PRO TIPS

### Tip 1: Use --limit for Testing
Always test with `--limit` flag first before running full pipeline.

### Tip 2: Run Analysis Overnight
AI analysis takes 30-40 hours. Start before bed or weekend.

### Tip 3: Monitor with Checkpoints
Check `data/cache/checkpoints/` to see progress. Latest file = current position.

### Tip 4: Resume if Interrupted
If analysis stops, use `--resume` flag to continue from checkpoint.

### Tip 5: Backup Before Major Changes
Always backup `reviews.db` before re-running analysis with `--force`.

### Tip 6: Use Comparison Tools
Run `sprint_comparison_dashboard.py` to verify improvements after each sprint.

---

## 🎯 CHECKLISTS

### ✅ Pre-Analysis Checklist:
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] .env file configured
- [ ] Data downloaded (Bước 2)
- [ ] Database created (Bước 3)
- [ ] Data parsed & loaded (Bước 4)

### ✅ Post-Analysis Checklist:
- [ ] No errors in analysis log
- [ ] Checkpoint files created
- [ ] aspect_sentiments table populated
- [ ] Confidence scores look reasonable (0.75+ avg)
- [ ] Aspects/review reasonable (1.0-2.0)

### ✅ Pre-Demo Checklist:
- [ ] Summaries generated (Bước 6)
- [ ] UI launches without errors
- [ ] Can navigate: Category → Brand → Product
- [ ] Charts display correctly
- [ ] Sample reviews shown
- [ ] Validation warnings visible

---

## 📞 SUPPORT

### If Pipeline Breaks:

1. **Check which step failed**
   - Look at error message
   - Check logs in `logs/` folder

2. **Common fixes**
   - Encoding error → Set `$env:PYTHONIOENCODING='utf-8'`
   - Database locked → Close all connections
   - Out of memory → Reduce batch size
   - CUDA error → Disable GPU or update drivers

3. **Reset pipeline**
   ```bash
   # Start over from specific step
   # Database setup
   python scripts/setup_database.py --reset
   
   # Re-parse data
   python scripts/parse_data.py --category electronics --force
   
   # Re-analyze
   python scripts/run_analysis.py --category electronics --force
   ```

---

## 🏆 SUCCESS CRITERIA

Pipeline is successful if:

✅ **Database populated:**
- 3,000 products with `is_selected=True`
- 2-3M reviews loaded
- 200-300 brands created

✅ **Analysis complete:**
- 3-4M aspect sentiments
- Avg confidence ≥ 0.75
- Aspects/review ≤ 2.0

✅ **Summaries generated:**
- 3,000 product summaries
- 287 brand summaries
- Representative reviews selected

✅ **UI working:**
- Can navigate all products
- Charts render correctly
- Reviews display with AI labels
- Validation warnings shown

---

## 🎉 CONGRATULATIONS!

If you've completed all steps, you now have:

- ✅ **2-3M reviews** analyzed by AI
- ✅ **90%+ accuracy** with improvements
- ✅ **Interactive dashboard** for insights
- ✅ **Production-ready system** for demo

**Next steps:**
- 📊 Explore insights in UI
- 💼 Add to portfolio
- 🎓 Present to professor/employer
- 🚀 Deploy to production (optional)

---

**Questions?** Check:
- `SPRINT_COMPLETE_SUMMARY.md` - Overall improvements
- `IMPROVEMENT_SUMMARY.md` - Detailed sprint plans
- `README.md` - Project overview
- `TROUBLESHOOTING.md` - Common issues

**Happy analyzing!** 🎊

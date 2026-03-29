-- Product Review Analyzer Database Schema
-- SQLite Database for flexible, scalable review analysis

-- Categories table
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) UNIQUE NOT NULL,
    amazon_id VARCHAR(100),
    config_path TEXT,
    total_brands INTEGER DEFAULT 0,
    total_products INTEGER DEFAULT 0,
    total_reviews INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Brands table (replaces stores for better business logic)
CREATE TABLE IF NOT EXISTS brands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
    normalized_name VARCHAR(255) NOT NULL,
    category_id INTEGER NOT NULL,
    product_count INTEGER DEFAULT 0,
    avg_rating FLOAT,
    total_reviews INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE,
    UNIQUE(normalized_name, category_id)
);

CREATE INDEX IF NOT EXISTS idx_brands_category ON brands(category_id);
CREATE INDEX IF NOT EXISTS idx_brands_normalized ON brands(normalized_name);

-- Products table
CREATE TABLE IF NOT EXISTS products (
    parent_asin VARCHAR(20) PRIMARY KEY,
    title TEXT NOT NULL,
    brand_id INTEGER,
    category_id INTEGER NOT NULL,
    average_rating FLOAT,
    rating_number INTEGER DEFAULT 0,
    price FLOAT,
    image_url TEXT,
    features TEXT,
    description TEXT,
    product_metadata TEXT,
    
    -- Selection flags
    is_selected BOOLEAN DEFAULT 0,
    selection_strategy VARCHAR(50),
    selection_bin VARCHAR(20),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (brand_id) REFERENCES brands(id) ON DELETE SET NULL,
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand_id);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_id);
CREATE INDEX IF NOT EXISTS idx_products_selected ON products(is_selected);
CREATE INDEX IF NOT EXISTS idx_products_rating ON products(rating_number DESC);

-- Reviews table
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_asin VARCHAR(20) NOT NULL,
    user_id VARCHAR(100),
    rating FLOAT NOT NULL,
    title TEXT,
    text TEXT,
    text_length INTEGER,
    timestamp INTEGER,
    verified_purchase BOOLEAN DEFAULT 0,
    helpful_vote INTEGER DEFAULT 0,
    has_images BOOLEAN DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_asin) REFERENCES products(parent_asin) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(parent_asin);
CREATE INDEX IF NOT EXISTS idx_reviews_length ON reviews(text_length);
CREATE INDEX IF NOT EXISTS idx_reviews_timestamp ON reviews(timestamp);
CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);

-- Aspect Sentiments table (AI analysis results)
CREATE TABLE IF NOT EXISTS aspect_sentiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER NOT NULL,
    aspect_name VARCHAR(50) NOT NULL,
    aspect_tier INTEGER NOT NULL,
    sentiment VARCHAR(10) NOT NULL,
    confidence_score FLOAT NOT NULL,
    detection_method VARCHAR(20) DEFAULT 'keyword',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (review_id) REFERENCES reviews(id) ON DELETE CASCADE,
    UNIQUE(review_id, aspect_name)
);

CREATE INDEX IF NOT EXISTS idx_aspects_review ON aspect_sentiments(review_id);
CREATE INDEX IF NOT EXISTS idx_aspects_aspect ON aspect_sentiments(aspect_name);
CREATE INDEX IF NOT EXISTS idx_aspects_sentiment ON aspect_sentiments(sentiment);
CREATE INDEX IF NOT EXISTS idx_aspects_tier ON aspect_sentiments(aspect_tier);

-- Product Summaries table (cached aggregations)
CREATE TABLE IF NOT EXISTS product_summaries (
    parent_asin VARCHAR(20) PRIMARY KEY,
    category_id INTEGER NOT NULL,
    
    -- Overall statistics
    total_reviews INTEGER DEFAULT 0,
    avg_rating FLOAT,
    rating_distribution TEXT,
    
    -- Overall sentiment distribution
    overall_positive INTEGER DEFAULT 0,
    overall_negative INTEGER DEFAULT 0,
    overall_neutral INTEGER DEFAULT 0,
    
    -- Aspect-based summary (JSON)
    aspects_summary TEXT,
    
    -- Representative reviews (JSON arrays of review IDs)
    top_positive_review_ids TEXT,
    top_negative_review_ids TEXT,
    top_mixed_review_ids TEXT,
    
    -- Metadata
    aspects_analyzed TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (parent_asin) REFERENCES products(parent_asin) ON DELETE CASCADE,
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_summary_category ON product_summaries(category_id);
CREATE INDEX IF NOT EXISTS idx_summary_updated ON product_summaries(last_updated);

-- Brand Summaries table (aggregated brand-level insights)
CREATE TABLE IF NOT EXISTS brand_summaries (
    brand_id INTEGER PRIMARY KEY,
    category_id INTEGER NOT NULL,
    
    total_products INTEGER DEFAULT 0,
    total_reviews INTEGER DEFAULT 0,
    avg_rating FLOAT,
    
    -- Aspect aggregation across all brand products (JSON)
    aspects_summary TEXT,
    
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (brand_id) REFERENCES brands(id) ON DELETE CASCADE,
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_brand_summary_category ON brand_summaries(category_id);

-- Processing Status table (track analysis progress)
CREATE TABLE IF NOT EXISTS processing_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id INTEGER NOT NULL,
    stage VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    progress FLOAT DEFAULT 0.0,
    total_items INTEGER,
    processed_items INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_processing_category ON processing_status(category_id);
CREATE INDEX IF NOT EXISTS idx_processing_status ON processing_status(status);

-- Analysis Cache table (cache individual review analysis for reuse)
CREATE TABLE IF NOT EXISTS analysis_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_text_hash VARCHAR(64) UNIQUE NOT NULL,
    aspect_name VARCHAR(50) NOT NULL,
    sentiment VARCHAR(10) NOT NULL,
    confidence_score FLOAT NOT NULL,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cache_hash ON analysis_cache(review_text_hash);
CREATE INDEX IF NOT EXISTS idx_cache_aspect ON analysis_cache(aspect_name);

-- Create views for common queries
CREATE VIEW IF NOT EXISTS v_product_details AS
SELECT 
    p.parent_asin,
    p.title,
    p.average_rating,
    p.rating_number,
    p.price,
    p.image_url,
    b.name as brand_name,
    c.name as category_name,
    p.is_selected
FROM products p
LEFT JOIN brands b ON p.brand_id = b.id
LEFT JOIN categories c ON p.category_id = c.id;

CREATE VIEW IF NOT EXISTS v_review_sentiments AS
SELECT 
    r.id as review_id,
    r.parent_asin,
    r.rating,
    r.text,
    r.title as review_title,
    r.verified_purchase,
    a.aspect_name,
    a.sentiment,
    a.confidence_score,
    a.aspect_tier
FROM reviews r
LEFT JOIN aspect_sentiments a ON r.id = a.review_id;

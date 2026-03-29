"""SQLAlchemy ORM models for the database."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, 
    ForeignKey, DateTime, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Category(Base):
    """Category model (e.g., Electronics, Beauty, Books)."""
    __tablename__ = 'categories'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    amazon_id = Column(String(100))
    config_path = Column(Text)
    total_brands = Column(Integer, default=0)
    total_products = Column(Integer, default=0)
    total_reviews = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    brands = relationship("Brand", back_populates="category", cascade="all, delete-orphan")
    products = relationship("Product", back_populates="category", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Category(name='{self.name}', products={self.total_products})>"


class Brand(Base):
    """Brand model (e.g., Apple, Samsung, Sony)."""
    __tablename__ = 'brands'
    __table_args__ = (
        Index('idx_brands_category', 'category_id'),
        Index('idx_brands_normalized', 'normalized_name'),
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=False)
    category_id = Column(Integer, ForeignKey('categories.id', ondelete='CASCADE'), nullable=False)
    product_count = Column(Integer, default=0)
    avg_rating = Column(Float)
    total_reviews = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    category = relationship("Category", back_populates="brands")
    products = relationship("Product", back_populates="brand")
    summary = relationship("BrandSummary", back_populates="brand", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Brand(name='{self.name}', products={self.product_count})>"


class Product(Base):
    """Product model."""
    __tablename__ = 'products'
    __table_args__ = (
        Index('idx_products_brand', 'brand_id'),
        Index('idx_products_category', 'category_id'),
        Index('idx_products_selected', 'is_selected'),
        Index('idx_products_rating', 'rating_number'),
    )
    
    parent_asin = Column(String(20), primary_key=True)
    title = Column(Text, nullable=False)
    brand_id = Column(Integer, ForeignKey('brands.id', ondelete='SET NULL'))
    category_id = Column(Integer, ForeignKey('categories.id', ondelete='CASCADE'), nullable=False)
    average_rating = Column(Float)
    rating_number = Column(Integer, default=0)
    price = Column(Float)
    image_url = Column(Text)
    features = Column(Text)  # JSON string
    description = Column(Text)
    product_metadata = Column(Text)  # Full metadata JSON (renamed from 'metadata' - reserved word)
    
    # Selection flags
    is_selected = Column(Boolean, default=False)
    selection_strategy = Column(String(50))
    selection_bin = Column(String(20))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    brand = relationship("Brand", back_populates="products")
    category = relationship("Category", back_populates="products")
    reviews = relationship("Review", back_populates="product", cascade="all, delete-orphan")
    summary = relationship("ProductSummary", back_populates="product", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Product(asin='{self.parent_asin}', title='{self.title[:30]}...')>"


class Review(Base):
    """Review model."""
    __tablename__ = 'reviews'
    __table_args__ = (
        Index('idx_reviews_product', 'parent_asin'),
        Index('idx_reviews_length', 'text_length'),
        Index('idx_reviews_timestamp', 'timestamp'),
        Index('idx_reviews_rating', 'rating'),
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    parent_asin = Column(String(20), ForeignKey('products.parent_asin', ondelete='CASCADE'), nullable=False)
    user_id = Column(String(100))
    rating = Column(Float, nullable=False)
    title = Column(Text)
    text = Column(Text)
    text_length = Column(Integer)
    timestamp = Column(Integer)
    verified_purchase = Column(Boolean, default=False)
    helpful_vote = Column(Integer, default=0)
    has_images = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="reviews")
    aspect_sentiments = relationship("AspectSentiment", back_populates="review", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Review(id={self.id}, product='{self.parent_asin}', rating={self.rating})>"


class AspectSentiment(Base):
    """Aspect-based sentiment analysis results."""
    __tablename__ = 'aspect_sentiments'
    __table_args__ = (
        Index('idx_aspects_review', 'review_id'),
        Index('idx_aspects_aspect', 'aspect_name'),
        Index('idx_aspects_sentiment', 'sentiment'),
        Index('idx_aspects_tier', 'aspect_tier'),
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(Integer, ForeignKey('reviews.id', ondelete='CASCADE'), nullable=False)
    aspect_name = Column(String(50), nullable=False)
    aspect_tier = Column(Integer, nullable=False)
    sentiment = Column(String(10), nullable=False)  # positive, negative, neutral
    confidence_score = Column(Float, nullable=False)
    detection_method = Column(String(20), default='keyword')  # keyword, ml, hybrid
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    review = relationship("Review", back_populates="aspect_sentiments")
    
    def __repr__(self):
        return f"<AspectSentiment(aspect='{self.aspect_name}', sentiment='{self.sentiment}')>"


class ProductSummary(Base):
    """Cached product-level summary."""
    __tablename__ = 'product_summaries'
    __table_args__ = (
        Index('idx_summary_category', 'category_id'),
        Index('idx_summary_updated', 'last_updated'),
    )
    
    parent_asin = Column(String(20), ForeignKey('products.parent_asin', ondelete='CASCADE'), primary_key=True)
    category_id = Column(Integer, ForeignKey('categories.id', ondelete='CASCADE'), nullable=False)
    
    # Overall statistics
    total_reviews = Column(Integer, default=0)
    avg_rating = Column(Float)
    rating_distribution = Column(Text)  # JSON
    
    # Overall sentiment
    overall_positive = Column(Integer, default=0)
    overall_negative = Column(Integer, default=0)
    overall_neutral = Column(Integer, default=0)
    
    # Aspect-based summary
    aspects_summary = Column(Text)  # JSON
    
    # Representative reviews
    top_positive_review_ids = Column(Text)  # JSON array
    top_negative_review_ids = Column(Text)  # JSON array
    top_mixed_review_ids = Column(Text)  # JSON array
    
    # Metadata
    aspects_analyzed = Column(Text)  # JSON array
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="summary")
    
    def __repr__(self):
        return f"<ProductSummary(asin='{self.parent_asin}', reviews={self.total_reviews})>"


class BrandSummary(Base):
    """Cached brand-level summary."""
    __tablename__ = 'brand_summaries'
    __table_args__ = (
        Index('idx_brand_summary_category', 'category_id'),
    )
    
    brand_id = Column(Integer, ForeignKey('brands.id', ondelete='CASCADE'), primary_key=True)
    category_id = Column(Integer, ForeignKey('categories.id', ondelete='CASCADE'), nullable=False)
    
    total_products = Column(Integer, default=0)
    total_reviews = Column(Integer, default=0)
    avg_rating = Column(Float)
    
    # Aspect aggregation
    aspects_summary = Column(Text)  # JSON
    
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    brand = relationship("Brand", back_populates="summary")
    
    def __repr__(self):
        return f"<BrandSummary(brand_id={self.brand_id}, products={self.total_products})>"


class ProcessingStatus(Base):
    """Track processing progress."""
    __tablename__ = 'processing_status'
    __table_args__ = (
        Index('idx_processing_category', 'category_id'),
        Index('idx_processing_status', 'status'),
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    category_id = Column(Integer, ForeignKey('categories.id', ondelete='CASCADE'), nullable=False)
    stage = Column(String(50), nullable=False)  # parsing, aspect_analysis, summarization
    status = Column(String(20), nullable=False, default='pending')  # pending, running, completed, failed
    progress = Column(Float, default=0.0)
    total_items = Column(Integer)
    processed_items = Column(Integer, default=0)
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    def __repr__(self):
        return f"<ProcessingStatus(stage='{self.stage}', status='{self.status}', progress={self.progress}%)>"


class AnalysisCache(Base):
    """Cache for AI analysis results."""
    __tablename__ = 'analysis_cache'
    __table_args__ = (
        Index('idx_cache_hash', 'review_text_hash'),
        Index('idx_cache_aspect', 'aspect_name'),
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    review_text_hash = Column(String(64), unique=True, nullable=False)
    aspect_name = Column(String(50), nullable=False)
    sentiment = Column(String(10), nullable=False)
    confidence_score = Column(Float, nullable=False)
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=1)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AnalysisCache(aspect='{self.aspect_name}', hits={self.access_count})>"

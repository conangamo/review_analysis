"""Main Streamlit application."""

import sys
from pathlib import Path
import json
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from sqlalchemy import desc

from src.core import get_env, ConfigLoader
from src.database.db_manager import DatabaseManager
from src.database.models import Category, Brand, Product, ProductSummary, Review, AspectSentiment
from src.ui.components.filters import (
    render_category_selector,
    render_brand_selector,
    render_product_selector
)
from src.ui.components.charts import (
    create_aspect_bar_chart,
    create_overall_sentiment_pie,
    create_rating_distribution_chart,
    create_aspect_comparison_chart
)
from src.ui.components.review_cards import (
    render_review_list,
    render_sentiment_breakdown_card
)
from src.ui.utils.formatters import format_number, format_rating, format_aspect_name


# Page configuration
st.set_page_config(
    page_title="Product Review Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def init_database():
    """Initialize database connection."""
    env = get_env()
    db_path = env.get_path("DB_PATH", "./data/processed/reviews.db")
    return DatabaseManager(str(db_path))


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_categories(_db_manager):
    """Load categories from database."""
    with _db_manager.get_session() as session:
        categories = session.query(Category).all()
        return [
            {
                'id': c.id,
                'name': c.name,
                'total_products': c.total_products,
                'total_reviews': c.total_reviews,
                'total_brands': c.total_brands
            }
            for c in categories
        ]


@st.cache_data(ttl=300)
def load_brands(_db_manager, category_id: int):
    """Load brands for a category."""
    with _db_manager.get_session() as session:
        brands = session.query(Brand).filter_by(
            category_id=category_id
        ).filter(
            Brand.product_count > 0
        ).order_by(
            desc(Brand.product_count)
        ).all()
        
        return [
            {
                'id': b.id,
                'name': b.name,
                'product_count': b.product_count,
                'avg_rating': b.avg_rating,
                'total_reviews': b.total_reviews
            }
            for b in brands
        ]


@st.cache_data(ttl=300)
def load_products(_db_manager, category_id: int, brand_id: int = None):
    """Load products for a category/brand."""
    with _db_manager.get_session() as session:
        query = session.query(Product).filter_by(
            category_id=category_id,
            is_selected=True
        )
        
        if brand_id:
            query = query.filter_by(brand_id=brand_id)
        
        products = query.order_by(desc(Product.rating_number)).all()
        
        return [
            {
                'parent_asin': p.parent_asin,
                'title': p.title,
                'average_rating': p.average_rating,
                'rating_number': p.rating_number,
                'price': p.price,
                'image_url': p.image_url
            }
            for p in products
        ]


@st.cache_data(ttl=300)
def load_product_summary(_db_manager, parent_asin: str):
    """Load product summary."""
    with _db_manager.get_session() as session:
        summary = session.query(ProductSummary).filter_by(
            parent_asin=parent_asin
        ).first()
        
        if not summary:
            return None
        
        # Parse JSON fields
        aspects_summary = json.loads(summary.aspects_summary) if summary.aspects_summary else {}
        rating_dist = json.loads(summary.rating_distribution) if summary.rating_distribution else {}
        
        return {
            'total_reviews': summary.total_reviews,
            'avg_rating': summary.avg_rating,
            'rating_distribution': rating_dist,
            'overall_positive': summary.overall_positive,
            'overall_negative': summary.overall_negative,
            'overall_neutral': summary.overall_neutral,
            'aspects_summary': aspects_summary,
            'top_positive_ids': json.loads(summary.top_positive_review_ids) if summary.top_positive_review_ids else [],
            'top_negative_ids': json.loads(summary.top_negative_review_ids) if summary.top_negative_review_ids else [],
            'top_mixed_ids': json.loads(summary.top_mixed_review_ids) if summary.top_mixed_review_ids else []
        }


def load_reviews_with_aspects(_db_manager, review_ids: List[int]):
    """Load reviews and their aspects."""
    if not review_ids:
        return [], {}
    
    with _db_manager.get_session() as session:
        reviews = session.query(Review).filter(
            Review.id.in_(review_ids)
        ).all()
        
        review_list = [
            {
                'id': r.id,
                'rating': r.rating,
                'title': r.title,
                'text': r.text,
                'timestamp': r.timestamp,
                'verified_purchase': r.verified_purchase,
                'helpful_vote': r.helpful_vote
            }
            for r in reviews
        ]
        
        # Load aspects for these reviews
        aspects = session.query(AspectSentiment).filter(
            AspectSentiment.review_id.in_(review_ids)
        ).all()
        
        # Group by review_id
        aspects_map = {}
        for aspect in aspects:
            review_id = aspect.review_id
            if review_id not in aspects_map:
                aspects_map[review_id] = []
            
            aspects_map[review_id].append({
                'aspect_name': aspect.aspect_name,
                'sentiment': aspect.sentiment,
                'confidence_score': aspect.confidence_score,
                'tier': aspect.aspect_tier
            })
        
        return review_list, aspects_map


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("📊 Product Review Analyzer")
    st.markdown("*AI-powered aspect-based sentiment analysis for product reviews*")
    st.markdown("---")
    
    # Initialize database
    try:
        db_manager = init_database()
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        st.info("Please run: `python scripts/setup_database.py`")
        return
    
    # Sidebar navigation
    st.sidebar.title("🔍 Navigation")
    
    # Step 1: Select category
    categories = load_categories(db_manager)
    
    if not categories:
        st.warning("⚠️  No categories found in database")
        st.info("Please run: `python scripts/parse_data.py --category electronics`")
        return
    
    selected_category = render_category_selector(categories)
    
    if not selected_category:
        st.info("👈 Select a category from the sidebar to begin")
        return
    
    # Step 2: Select brand
    brands = load_brands(db_manager, selected_category['id'])
    
    if not brands:
        st.warning(f"⚠️  No brands found for {selected_category['name']}")
        return
    
    selected_brand = render_brand_selector(brands, enable_search=True)
    
    if not selected_brand:
        st.info("👈 Select a brand to continue")
        return
    
    # Step 3: Select product
    products = load_products(
        db_manager,
        selected_category['id'],
        selected_brand['id']
    )
    
    if not products:
        st.warning(f"⚠️  No products found for {selected_brand['name']}")
        return
    
    selected_product = render_product_selector(products, enable_search=True)
    
    if not selected_product:
        st.info("👈 Select a product to analyze")
        return
    
    # Main content area
    display_product_analysis(db_manager, selected_product)


def display_product_analysis(db_manager, product: Dict[str, Any]):
    """
    Display product analysis.
    
    Args:
        db_manager: Database manager
        product: Product dictionary
    """
    parent_asin = product['parent_asin']
    
    # Load summary
    summary = load_product_summary(db_manager, parent_asin)
    
    if not summary:
        st.warning("⚠️  No analysis data available for this product")
        st.info(f"Run: `python scripts/run_analysis.py --category electronics`")
        return
    
    # Product header
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(product['title'])
        
        if product.get('price'):
            st.markdown(f"**Price:** ${product['price']:.2f}")
    
    with col2:
        if product.get('image_url'):
            st.image(product['image_url'], width=200)
    
    st.markdown("---")
    
    # PRIORITY 1: Basic Statistics
    st.markdown("## 📈 Product Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Average Rating",
            value=format_rating(summary['avg_rating'])
        )
    
    with col2:
        st.metric(
            label="Total Reviews",
            value=format_number(summary['total_reviews'])
        )
    
    with col3:
        # Calculate overall positive percentage
        total_sentiment = (
            summary['overall_positive'] +
            summary['overall_negative'] +
            summary['overall_neutral']
        )
        pos_pct = (summary['overall_positive'] / total_sentiment * 100) if total_sentiment > 0 else 0
        st.metric(
            label="Positive Sentiment",
            value=f"{pos_pct:.0f}%"
        )
    
    st.markdown("---")
    
    # Overall sentiment breakdown
    render_sentiment_breakdown_card(summary)
    
    st.markdown("---")
    
    # PRIORITY 2: Aspect-Based Analysis
    st.markdown("## 🎯 Aspect-Based Analysis")
    
    aspects_summary = summary.get('aspects_summary', {})
    
    if not aspects_summary:
        st.warning("No aspect analysis available")
    else:
        # Show comparison chart
        st.markdown("### Aspect Comparison")
        comparison_chart = create_aspect_comparison_chart(aspects_summary, top_n=10)
        st.plotly_chart(comparison_chart, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed aspect breakdown
        st.markdown("### Detailed Breakdown")
        
        # Sort aspects by tier and mentions
        sorted_aspects = sorted(
            aspects_summary.items(),
            key=lambda x: x[1].get('total_mentions', 0),
            reverse=True
        )
        
        # Display each aspect
        for aspect_name, aspect_data in sorted_aspects:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Aspect bar chart
                chart = create_aspect_bar_chart(aspect_name, aspect_data, height=80)
                st.plotly_chart(chart, use_container_width=True)
            
            with col2:
                # Stats
                st.markdown(f"**{format_aspect_name(aspect_name)}**")
                st.caption(f"Mentions: {aspect_data['total_mentions']:,}")
                st.caption(f"Avg confidence: {aspect_data.get('avg_confidence', 0):.0%}")
    
    st.markdown("---")
    
    # PRIORITY 3: Sample Reviews
    st.markdown("## 💬 Representative Reviews")
    
    # Tabs for different sentiment types
    tab1, tab2, tab3 = st.tabs(["🟢 Positive", "🔴 Negative", "⚪ Mixed"])
    
    with tab1:
        review_ids = summary.get('top_positive_ids', [])
        if review_ids:
            reviews, aspects_map = load_reviews_with_aspects(db_manager, review_ids)
            render_review_list(reviews, aspects_map, max_reviews=5)
        else:
            st.info("No positive reviews available")
    
    with tab2:
        review_ids = summary.get('top_negative_ids', [])
        if review_ids:
            reviews, aspects_map = load_reviews_with_aspects(db_manager, review_ids)
            render_review_list(reviews, aspects_map, max_reviews=5)
        else:
            st.info("No negative reviews available")
    
    with tab3:
        review_ids = summary.get('top_mixed_ids', [])
        if review_ids:
            reviews, aspects_map = load_reviews_with_aspects(db_manager, review_ids)
            render_review_list(reviews, aspects_map, max_reviews=5)
        else:
            st.info("No mixed reviews available")


# Sidebar footer
def render_sidebar_footer():
    """Render sidebar footer with info."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This tool uses AI to analyze product reviews "
        "and extract sentiment for specific aspects like "
        "battery, screen, performance, etc."
    )
    
    st.sidebar.markdown("### 🛠️ Tools")
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# Run app
if __name__ == "__main__":
    try:
        main()
        render_sidebar_footer()
    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        st.exception(e)

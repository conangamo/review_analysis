"""Main Streamlit application."""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from src.core import get_env
from src.database.db_manager import DatabaseManager
from src.ui.components.filters import render_category_selector, render_brand_selector, render_product_selector
from src.ui.components.charts import create_aspect_bar_chart, create_aspect_comparison_chart
from src.ui.components.review_cards import render_review_list, render_sentiment_breakdown_card
from src.ui.services import (
    get_analysis_progress,
    get_brands,
    get_categories,
    get_product_summary,
    get_products,
    get_reviews_with_aspects,
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
    return get_categories(_db_manager)


@st.cache_data(ttl=300)
def load_brands(_db_manager, category_id: int):
    """Load brands for a category."""
    return get_brands(_db_manager, category_id)


@st.cache_data(ttl=300)
def load_products(_db_manager, category_id: int, brand_id: int = None):
    """Load products for a category/brand that have aspect analysis data."""
    return get_products(_db_manager, category_id, brand_id)


@st.cache_data(ttl=300)
def load_product_summary(_db_manager, parent_asin: str):
    """Load product summary."""
    return get_product_summary(_db_manager, parent_asin)


def load_reviews_with_aspects(_db_manager, review_ids: List[int]):
    """Load reviews and their aspects."""
    return get_reviews_with_aspects(_db_manager, review_ids)


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
    
    # Show analysis status
    analyzed_products, total_products = get_analysis_progress(db_manager, selected_category['id'])
    if analyzed_products > 0:
        st.sidebar.success(f"✅ {analyzed_products:,} / {total_products:,} products analyzed")
    else:
        st.sidebar.warning(f"⚠️ 0 / {total_products:,} products analyzed")
        st.sidebar.info("Run analysis pipeline first")
    
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
        st.warning(f"⚠️  No analyzed products found for {selected_brand['name']}")
        st.info(
            f"**Products found but not yet analyzed.**\n\n"
            f"To analyze products, run:\n\n"
            f"```bash\n"
            f"python scripts/run_analysis.py --category electronics --limit 1000\n"
            f"python scripts/generate_summaries.py --category electronics\n"
            f"```"
        )
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

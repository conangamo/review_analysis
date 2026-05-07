"""Filter components for UI navigation."""

import streamlit as st
from typing import List, Dict, Any, Optional


def render_category_selector(categories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Render category selector.
    
    Args:
        categories: List of category dictionaries
    
    Returns:
        Selected category or None
    """
    if not categories:
        st.sidebar.warning("No categories available")
        return None
    
    # Format options
    options = {f"{cat['name']} ({cat['total_products']:,} products)": cat for cat in categories}
    
    selected_key = st.sidebar.selectbox(
        "📁 Select Category",
        options=list(options.keys()),
        key='category_selector'
    )
    
    return options[selected_key] if selected_key else None


def render_brand_selector(
    brands: List[Dict[str, Any]],
    enable_search: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Render brand selector with optional search.
    
    Args:
        brands: List of brand dictionaries
        enable_search: Enable search functionality
    
    Returns:
        Selected brand or None
    """
    if not brands:
        st.sidebar.warning("No brands available")
        return None
    
    # Search functionality
    if enable_search and len(brands) > 10:
        search_term = st.sidebar.text_input(
            "🔍 Search brands",
            key='brand_search',
            placeholder="Type to filter..."
        )
        
        if search_term:
            brands = [
                b for b in brands
                if search_term.lower() in b['name'].lower()
            ]
    
    if not brands:
        st.sidebar.info("No brands match your search")
        return None
    
    # Format options
    options = {}
    for brand in brands:
        avg_rating = brand.get('avg_rating') or 0
        analyzed_products = brand.get("analyzed_products", 0)
        analysis_icon = "✅" if brand.get("has_analysis") else "⏳"
        label = (
            f"{analysis_icon} {brand['name']} "
            f"({brand['product_count']} products, AI:{analyzed_products}, ⭐{avg_rating:.1f})"
        )
        options[label] = brand
    
    selected_key = st.sidebar.selectbox(
        "🏷️  Select Brand",
        options=list(options.keys()),
        key='brand_selector'
    )
    
    return options[selected_key] if selected_key else None


def render_product_selector(
    products: List[Dict[str, Any]],
    enable_search: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Render product selector with optional search.
    
    Args:
        products: List of product dictionaries
        enable_search: Enable search functionality
    
    Returns:
        Selected product or None
    """
    if not products:
        st.sidebar.warning("No products available")
        return None
    
    # Search functionality
    if enable_search and len(products) > 10:
        search_term = st.sidebar.text_input(
            "🔍 Search products",
            key='product_search',
            placeholder="Type to filter..."
        )
        
        if search_term:
            products = [
                p for p in products
                if search_term.lower() in p['title'].lower()
            ]
    
    if not products:
        st.sidebar.info("No products match your search")
        return None
    
    # Format options with truncated titles
    options = {}
    for product in products:
        title = product['title']
        if len(title) > 60:
            title = title[:57] + "..."
        
        avg_rating = product.get('average_rating') or 0
        rating_number = product.get('rating_number') or 0
        
        analysis_icon = "✅" if product.get("has_analysis") else "⏳"
        label = f"{analysis_icon} {title} (⭐{avg_rating:.1f}, {rating_number:,} reviews)"
        options[label] = product
    
    selected_key = st.sidebar.selectbox(
        "📦 Select Product",
        options=list(options.keys()),
        key='product_selector'
    )
    
    return options[selected_key] if selected_key else None


def render_filters_panel(
    show_rating_filter: bool = True,
    show_verified_filter: bool = True,
    show_sort: bool = True
) -> Dict[str, Any]:
    """
    Render filters panel with various options.
    
    Args:
        show_rating_filter: Show rating filter
        show_verified_filter: Show verified purchase filter
        show_sort: Show sort options
    
    Returns:
        Dictionary with filter values
    """
    filters = {}
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Filters")
    
    if show_rating_filter:
        rating_range = st.sidebar.slider(
            "Rating Range",
            min_value=1.0,
            max_value=5.0,
            value=(1.0, 5.0),
            step=0.5,
            key='rating_filter'
        )
        filters['rating_min'] = rating_range[0]
        filters['rating_max'] = rating_range[1]
    
    if show_verified_filter:
        verified_only = st.sidebar.checkbox(
            "Verified purchases only",
            value=False,
            key='verified_filter'
        )
        filters['verified_only'] = verified_only
    
    if show_sort:
        sort_options = {
            'Most Recent': 'timestamp_desc',
            'Highest Rating': 'rating_desc',
            'Lowest Rating': 'rating_asc',
            'Most Helpful': 'helpful_desc'
        }
        
        sort_by = st.sidebar.selectbox(
            "Sort By",
            options=list(sort_options.keys()),
            key='sort_filter'
        )
        filters['sort_by'] = sort_options[sort_by]
    
    return filters


# Example usage
if __name__ == "__main__":
    print("Filter components for Streamlit UI")
    print("\nComponents available:")
    print("  - render_category_selector()")
    print("  - render_brand_selector()")
    print("  - render_product_selector()")
    print("  - render_filters_panel()")
    print("\nUse these in src/ui/app.py")

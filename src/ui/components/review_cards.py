"""Review card components."""

import streamlit as st
import json
from typing import List, Dict, Any, Optional

try:
    from ..utils.formatters import (
        format_timestamp,
        format_confidence,
        get_sentiment_emoji,
        format_aspect_name
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.ui.utils.formatters import (
        format_timestamp,
        format_confidence,
        get_sentiment_emoji,
        format_aspect_name
    )


def _prepare_aspects_for_card(aspects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize sentiment, dedupe by aspect_name (keep highest confidence),
    and sort for stable display (tier, then name).
    """
    if not aspects:
        return []

    best: Dict[str, Dict[str, Any]] = {}
    for raw in aspects:
        name = (raw.get("aspect_name") or "").strip()
        if not name:
            continue
        sent = (raw.get("sentiment") or "neutral").strip().lower()
        if sent not in ("positive", "negative", "neutral"):
            sent = "neutral"
        try:
            conf = float(raw.get("confidence_score") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        tier = raw.get("tier")
        if tier is None:
            tier = raw.get("aspect_tier")

        prev = best.get(name)
        if prev is None or conf > float(prev.get("confidence_score") or 0.0):
            merged: Dict[str, Any] = {
                "aspect_name": name,
                "sentiment": sent,
                "confidence_score": conf,
                "tier": tier,
            }
            if raw.get("negation_adjusted"):
                merged["negation_adjusted"] = True
            if raw.get("original_sentiment"):
                merged["original_sentiment"] = raw["original_sentiment"]
            best[name] = merged

    out = list(best.values())
    out.sort(key=lambda x: (x.get("tier") if x.get("tier") is not None else 99, x["aspect_name"]))
    return out


def render_review_card(
    review: Dict[str, Any],
    aspects: List[Dict[str, Any]],
    show_confidence: bool = True,
    show_validation: bool = True  # 🔧 SPRINT 3: Show validation warnings
):
    """
    Render a review card with AI-detected aspects and validation warnings.
    
    Args:
        review: Review dictionary
        aspects: List of aspect sentiment dictionaries
        show_confidence: Show confidence scores
        show_validation: Show validation warnings (Sprint 3)
    """
    # Normalize / dedupe aspects so All reviews matches Representative data shape
    aspects = _prepare_aspects_for_card(aspects or [])

    # 🔧 SPRINT 3: Check for validation warnings
    has_warnings = False
    warning_badges = []
    
    # Check for rating-sentiment mismatch
    if show_validation and aspects:
        rating = review.get('rating', 3.0)
        positive_count = sum(1 for a in aspects if a.get('sentiment') == 'positive')
        negative_count = sum(1 for a in aspects if a.get('sentiment') == 'negative')
        total_count = len(aspects)
        
        if total_count > 0:
            negative_pct = negative_count / total_count
            positive_pct = positive_count / total_count
            
            # High rating but mostly negative
            if rating >= 4.0 and negative_pct >= 0.7:
                has_warnings = True
                warning_badges.append("⚠️ High rating + negative sentiment")
            
            # Low rating but mostly positive
            elif rating <= 2.0 and positive_pct >= 0.7:
                has_warnings = True
                warning_badges.append("⚠️ Low rating + positive sentiment")
        
        # Check for negation adjustments
        negation_count = sum(1 for a in aspects if a.get('negation_adjusted', False))
        if negation_count > 0:
            warning_badges.append(f"🔄 {negation_count} negation adjusted")
    
    # Create expander with rating, title, and warning badges
    rating_stars = "⭐" * int(review.get('rating', 0))
    title = review.get('title', 'No title')
    
    # Add warning indicator to title if needed
    title_prefix = "⚠️ " if has_warnings else ""
    
    with st.expander(f"{title_prefix}{rating_stars} {review['rating']:.1f} - {title}"):
        # 🔧 SPRINT 3: Show validation warnings first
        if warning_badges:
            for badge in warning_badges:
                st.warning(badge)
            st.markdown("---")
        
        # Review text
        st.markdown(f"**Review:**")
        st.write(review.get('text', 'No text'))
        
        # Metadata
        col1, col2, col3 = st.columns(3)
        
        with col1:
            timestamp = review.get('timestamp')
            if timestamp:
                st.caption(f"📅 {format_timestamp(timestamp)}")
        
        with col2:
            if review.get('verified_purchase'):
                st.caption("✅ Verified Purchase")
        
        with col3:
            helpful = review.get('helpful_vote', 0)
            if helpful > 0:
                st.caption(f"👍 {helpful} helpful")
        
        # AI-detected aspects (same block for Representative and All reviews)
        st.markdown("---")
        st.markdown("**🤖 AI Detected Aspects:**")
        if aspects:
            # Group by sentiment
            by_sentiment: Dict[str, List[Dict[str, Any]]] = {"positive": [], "negative": [], "neutral": []}
            for aspect in aspects:
                sent = aspect.get("sentiment") or "neutral"
                if sent not in by_sentiment:
                    sent = "neutral"
                by_sentiment[sent].append(aspect)

            # Display grouped
            for sentiment_type in ["positive", "negative", "neutral"]:
                aspect_list = by_sentiment[sentiment_type]
                if aspect_list:
                    emoji = get_sentiment_emoji(sentiment_type)

                    aspect_texts = []
                    for aspect in aspect_list:
                        aspect_text = format_aspect_name(aspect["aspect_name"])

                        # 🔧 SPRINT 3: Show confidence and validation info
                        info_parts = []

                        if show_confidence:
                            conf = format_confidence(aspect["confidence_score"])
                            info_parts.append(conf)

                        # Show tier
                        if aspect.get("tier") is not None:
                            info_parts.append(f"T{aspect['tier']}")

                        # Show if negation adjusted
                        if aspect.get("negation_adjusted", False):
                            info_parts.append("adj")

                        if info_parts:
                            aspect_text += f" ({', '.join(info_parts)})"

                        aspect_texts.append(aspect_text)

                    st.markdown(
                        f"{emoji} **{sentiment_type.title()}**: {', '.join(aspect_texts)}"
                    )
        else:
            st.caption(
                "Chưa có aspect AI cho review này — review chưa được phân tích hoặc "
                "không đạt ngưỡng keyword/confidence."
            )


def render_review_list(
    reviews: List[Dict[str, Any]],
    aspects_map: Dict[int, List[Dict[str, Any]]],
    max_reviews: Optional[int] = 10,
    show_confidence: bool = True,
    list_title: Optional[str] = None,
    list_caption: Optional[str] = None,
):
    """
    Render a list of review cards.
    
    Args:
        reviews: List of review dictionaries
        aspects_map: Dictionary mapping review_id to aspects
        max_reviews: Maximum number of reviews to display; None = show all in ``reviews``
        show_confidence: Show confidence scores
        list_title: Optional markdown heading (e.g. paginated section title)
        list_caption: Optional caption below the heading
    """
    if not reviews:
        st.info("No reviews to display")
        return
    
    if max_reviews is None:
        display_reviews = reviews
    else:
        display_reviews = reviews[:max_reviews]

    if list_title:
        st.markdown(list_title)
    else:
        shown = len(display_reviews)
        total_in_batch = len(reviews)
        st.markdown(f"### 💬 Sample Reviews ({shown:,} of {total_in_batch:,})")

    if list_caption:
        st.caption(list_caption)
    
    for review in display_reviews:
        review_id = review.get("id")
        aspects = aspects_map.get(review_id, [])
        if not aspects and review_id is not None:
            aspects = aspects_map.get(int(review_id), [])
        render_review_card(review, aspects, show_confidence)
    
    if max_reviews is not None and len(reviews) > max_reviews:
        st.info(f"Showing {max_reviews:,} of {len(reviews):,} reviews. Adjust max_reviews to see more.")


def format_aspect_name(aspect: str) -> str:
    """Format aspect name for display."""
    return aspect.replace('_', ' ').title()


def render_sentiment_breakdown_card(summary_data: Dict[str, Any]):
    """
    Render sentiment breakdown card.
    
    Args:
        summary_data: Product summary data
    """
    total = summary_data.get('total_reviews', 0)
    positive = summary_data.get('overall_positive', 0)
    negative = summary_data.get('overall_negative', 0)
    neutral = summary_data.get('overall_neutral', 0)
    
    st.markdown("### 📊 Overall Sentiment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pos_pct = (positive / total * 100) if total > 0 else 0
        st.metric(
            label="🟢 Positive",
            value=f"{pos_pct:.1f}%",
            delta=f"{positive:,} reviews"
        )
    
    with col2:
        neg_pct = (negative / total * 100) if total > 0 else 0
        st.metric(
            label="🔴 Negative",
            value=f"{neg_pct:.1f}%",
            delta=f"{negative:,} reviews"
        )
    
    with col3:
        neu_pct = (neutral / total * 100) if total > 0 else 0
        st.metric(
            label="⚪ Neutral",
            value=f"{neu_pct:.1f}%",
            delta=f"{neutral:,} reviews"
        )


# Example usage
if __name__ == "__main__":
    print("Review card components for Streamlit UI")
    print("\nComponents available:")
    print("  - render_review_card()")
    print("  - render_review_list()")
    print("  - render_sentiment_breakdown_card()")
    print("\nUse these in src/ui/app.py")

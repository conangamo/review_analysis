"""Data access helpers for Streamlit UI."""

import json
from typing import Any, Dict, List, Tuple

from sqlalchemy import desc

from src.database.models import (
    AspectSentiment,
    Brand,
    Category,
    Product,
    ProductSummary,
    Review,
)


def get_categories(db_manager) -> List[Dict[str, Any]]:
    """Load categories from database."""
    with db_manager.get_session() as session:
        categories = session.query(Category).all()
        return [
            {
                "id": c.id,
                "name": c.name,
                "total_products": c.total_products,
                "total_reviews": c.total_reviews,
                "total_brands": c.total_brands,
            }
            for c in categories
        ]


def get_brands(db_manager, category_id: int) -> List[Dict[str, Any]]:
    """Load brands for a category."""
    with db_manager.get_session() as session:
        brands = (
            session.query(Brand)
            .filter_by(category_id=category_id)
            .filter(Brand.product_count > 0)
            .order_by(desc(Brand.product_count))
            .all()
        )

        return [
            {
                "id": b.id,
                "name": b.name,
                "product_count": b.product_count,
                "avg_rating": b.avg_rating,
                "total_reviews": b.total_reviews,
            }
            for b in brands
        ]


def get_products(db_manager, category_id: int, brand_id: int = None) -> List[Dict[str, Any]]:
    """Load products that have summary and sentiment data."""
    with db_manager.get_session() as session:
        query = (
            session.query(Product)
            .filter_by(category_id=category_id, is_selected=True)
            .join(ProductSummary, Product.parent_asin == ProductSummary.parent_asin)
            .filter(
                ProductSummary.overall_positive
                + ProductSummary.overall_negative
                + ProductSummary.overall_neutral
                > 0
            )
        )

        if brand_id:
            query = query.filter(Product.brand_id == brand_id)

        products = query.order_by(desc(Product.rating_number)).all()

        return [
            {
                "parent_asin": p.parent_asin,
                "title": p.title,
                "average_rating": p.average_rating,
                "rating_number": p.rating_number,
                "price": p.price,
                "image_url": p.image_url,
            }
            for p in products
        ]


def get_product_summary(db_manager, parent_asin: str) -> Dict[str, Any]:
    """Load product summary for display."""
    with db_manager.get_session() as session:
        summary = session.query(ProductSummary).filter_by(parent_asin=parent_asin).first()

        if not summary:
            return None

        return {
            "total_reviews": summary.total_reviews,
            "avg_rating": summary.avg_rating,
            "rating_distribution": json.loads(summary.rating_distribution)
            if summary.rating_distribution
            else {},
            "overall_positive": summary.overall_positive,
            "overall_negative": summary.overall_negative,
            "overall_neutral": summary.overall_neutral,
            "aspects_summary": json.loads(summary.aspects_summary) if summary.aspects_summary else {},
            "top_positive_ids": json.loads(summary.top_positive_review_ids)
            if summary.top_positive_review_ids
            else [],
            "top_negative_ids": json.loads(summary.top_negative_review_ids)
            if summary.top_negative_review_ids
            else [],
            "top_mixed_ids": json.loads(summary.top_mixed_review_ids)
            if summary.top_mixed_review_ids
            else [],
        }


def get_reviews_with_aspects(
    db_manager, review_ids: List[int]
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    """Load reviews with aspect details."""
    if not review_ids:
        return [], {}

    with db_manager.get_session() as session:
        reviews = session.query(Review).filter(Review.id.in_(review_ids)).all()
        review_list = [
            {
                "id": r.id,
                "rating": r.rating,
                "title": r.title,
                "text": r.text,
                "timestamp": r.timestamp,
                "verified_purchase": r.verified_purchase,
                "helpful_vote": r.helpful_vote,
            }
            for r in reviews
        ]

        aspects = session.query(AspectSentiment).filter(AspectSentiment.review_id.in_(review_ids)).all()
        aspects_map: Dict[int, List[Dict[str, Any]]] = {}
        for aspect in aspects:
            if aspect.review_id not in aspects_map:
                aspects_map[aspect.review_id] = []
            aspects_map[aspect.review_id].append(
                {
                    "aspect_name": aspect.aspect_name,
                    "sentiment": aspect.sentiment,
                    "confidence_score": aspect.confidence_score,
                    "tier": aspect.aspect_tier,
                }
            )

        return review_list, aspects_map


def get_analysis_progress(db_manager, category_id: int) -> Tuple[int, int]:
    """Return analyzed/total selected products for category."""
    with db_manager.get_session() as session:
        total_products = session.query(Product).filter_by(
            category_id=category_id,
            is_selected=True,
        ).count()

        analyzed_products = (
            session.query(Product)
            .filter_by(category_id=category_id, is_selected=True)
            .join(ProductSummary, Product.parent_asin == ProductSummary.parent_asin)
            .filter(
                ProductSummary.overall_positive
                + ProductSummary.overall_negative
                + ProductSummary.overall_neutral
                > 0
            )
            .count()
        )

    return analyzed_products, total_products

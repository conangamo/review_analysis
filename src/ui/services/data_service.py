"""Data access helpers for Streamlit UI."""

import json
from typing import Any, Dict, List, Tuple

from sqlalchemy import desc, func

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

        brand_ids = [b.id for b in brands]
        analyzed_map = {}
        if brand_ids:
            analyzed_rows = (
                session.query(
                    Product.brand_id,
                    func.count(func.distinct(Product.parent_asin)),
                )
                .join(ProductSummary, Product.parent_asin == ProductSummary.parent_asin)
                .filter(Product.brand_id.in_(brand_ids))
                .filter(
                    ProductSummary.overall_positive
                    + ProductSummary.overall_negative
                    + ProductSummary.overall_neutral
                    > 0
                )
                .group_by(Product.brand_id)
                .all()
            )
            analyzed_map = {brand_id: count for brand_id, count in analyzed_rows}

        brand_rows = []
        for b in brands:
            analyzed_products = analyzed_map.get(b.id, 0)
            brand_rows.append(
                {
                    "id": b.id,
                    "name": b.name,
                    "product_count": b.product_count,
                    "avg_rating": b.avg_rating,
                    "total_reviews": b.total_reviews,
                    "analyzed_products": analyzed_products,
                    "has_analysis": analyzed_products > 0,
                }
            )

        # Prioritize brands that already have analyzed products.
        brand_rows.sort(
            key=lambda item: (item["has_analysis"], item.get("analyzed_products", 0), item.get("product_count", 0)),
            reverse=True,
        )
        return brand_rows


def get_products(db_manager, category_id: int, brand_id: int = None) -> List[Dict[str, Any]]:
    """Load selected products and mark whether analysis is available."""
    with db_manager.get_session() as session:
        query = session.query(Product).filter_by(category_id=category_id, is_selected=True)

        if brand_id:
            query = query.filter(Product.brand_id == brand_id)

        products = query.order_by(desc(Product.rating_number)).all()
        if not products:
            return []

        asins = [p.parent_asin for p in products]
        summaries = (
            session.query(ProductSummary)
            .filter(ProductSummary.parent_asin.in_(asins))
            .all()
        )
        summary_map = {s.parent_asin: s for s in summaries}

        product_rows = []
        for p in products:
            summary = summary_map.get(p.parent_asin)
            has_analysis = False
            if summary:
                has_analysis = (
                    (summary.overall_positive or 0)
                    + (summary.overall_negative or 0)
                    + (summary.overall_neutral or 0)
                    > 0
                )

            product_rows.append(
                {
                "parent_asin": p.parent_asin,
                "title": p.title,
                "average_rating": p.average_rating,
                "rating_number": p.rating_number,
                "price": p.price,
                "image_url": p.image_url,
                "has_analysis": has_analysis,
            }
            )

        # Keep analyzed products first, then by review volume.
        product_rows.sort(
            key=lambda item: (item["has_analysis"], item.get("rating_number") or 0),
            reverse=True,
        )
        return product_rows


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

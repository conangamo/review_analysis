"""Print recently analyzed reviews with aspect sentiments from SQLite (read-only)."""

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import get_env  # noqa: E402


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_review_ids(
    conn: sqlite3.Connection,
    limit_reviews: int,
    random_order: bool,
    sentiment: Optional[str],
) -> List[int]:
    """Distinct review IDs (newest analysis or random)."""
    if random_order:
        where = "WHERE sentiment = ?" if sentiment else ""
        params = (sentiment,) if sentiment else ()
        cur = conn.execute(
            f"""
            SELECT DISTINCT review_id
            FROM aspect_sentiments
            {where}
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (*params, limit_reviews),
        )
        return [row[0] for row in cur.fetchall()]

    # Newest by latest aspect_sentiments.id per review
    if sentiment:
        cur = conn.execute(
            """
            SELECT review_id
            FROM aspect_sentiments
            WHERE sentiment = ?
            GROUP BY review_id
            ORDER BY MAX(id) DESC
            LIMIT ?
            """,
            (sentiment, limit_reviews),
        )
    else:
        cur = conn.execute(
            """
            SELECT review_id
            FROM aspect_sentiments
            GROUP BY review_id
            ORDER BY MAX(id) DESC
            LIMIT ?
            """,
            (limit_reviews,),
        )
    return [row[0] for row in cur.fetchall()]


def _print_review(
    conn: sqlite3.Connection,
    review_id: int,
    text_max: int,
    sentiment_filter: Optional[str],
) -> None:
    row = conn.execute(
        "SELECT id, parent_asin, rating, title, text FROM reviews WHERE id = ?",
        (review_id,),
    ).fetchone()
    if not row:
        print(f"  [missing review id={review_id}]")
        return

    text = (row["text"] or "").strip()
    preview = text[:text_max] + ("…" if len(text) > text_max else "")

    print("-" * 70)
    print(f"review_id={row['id']}  parent_asin={row['parent_asin']}  rating={row['rating']}")
    if row["title"]:
        print(f"title: {row['title'][:120]}")
    print(f"text: {preview}")

    q = "SELECT aspect_name, aspect_tier, sentiment, confidence_score, detection_method FROM aspect_sentiments WHERE review_id = ?"
    args: list = [review_id]
    if sentiment_filter:
        q += " AND sentiment = ?"
        args.append(sentiment_filter)

    q += " ORDER BY aspect_tier, aspect_name"
    aspects = conn.execute(q, args).fetchall()
    if not aspects:
        print("  (no aspect rows for this review with current filter)")
        return

    for a in aspects:
        print(
            f"  • {a['aspect_name']:20s}  {a['sentiment']:8s}  "
            f"conf={a['confidence_score']:.3f}  tier={a['aspect_tier']}  "
            f"method={a['detection_method'] or '-'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect reviews + aspect_sentiments stored in SQLite (read-only)."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="SQLite path (default: DB_PATH from .env / ./data/processed/reviews.db)",
    )
    parser.add_argument(
        "--limit-reviews",
        type=int,
        default=10,
        help="Max distinct reviews to print (default: 10)",
    )
    parser.add_argument(
        "--review-id",
        type=int,
        default=None,
        help="Inspect a single review by id",
    )
    parser.add_argument(
        "--sentiment",
        type=str,
        default=None,
        choices=["positive", "negative", "neutral"],
        help="Only show aspect rows (and sampling) matching this sentiment",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sample reviews randomly instead of newest-by-aspect-row",
    )
    parser.add_argument(
        "--text-max",
        type=int,
        default=400,
        help="Max chars of review text preview (default: 400)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print sentiment counts then exit",
    )
    args = parser.parse_args()

    env = get_env()
    db_path = Path(args.db_path) if args.db_path else env.get_path("DB_PATH", "./data/processed/reviews.db")

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1

    conn = _connect(db_path)

    if args.stats:
        rows = conn.execute(
            """
            SELECT sentiment, COUNT(*) AS c
            FROM aspect_sentiments
            GROUP BY sentiment
            ORDER BY c DESC
            """
        ).fetchall()
        print("\n📊 aspect_sentiments by sentiment:")
        total = sum(r["c"] for r in rows)
        for r in rows:
            pct = r["c"] / total * 100 if total else 0
            print(f"  {r['sentiment']:10s}  {r['c']:6d}  ({pct:.2f}%)")
        print(f"  TOTAL      {total:6d}")
        conn.close()
        return 0

    print("\n" + "=" * 70)
    print("🔎 Recent / sampled analyzed reviews")
    print("=" * 70)
    print(f"DB: {db_path}")

    if args.review_id is not None:
        _print_review(conn, args.review_id, args.text_max, args.sentiment)
        conn.close()
        print()
        return 0

    review_ids = _fetch_review_ids(
        conn, args.limit_reviews, args.random, args.sentiment
    )
    if not review_ids:
        print("No aspect_sentiments rows found (run analysis pipeline first).")
        conn.close()
        return 0

    print(f"Showing {len(review_ids)} review(s).\n")

    for rid in review_ids:
        _print_review(conn, rid, args.text_max, args.sentiment)

    conn.close()
    print("\n✅ Done (read-only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

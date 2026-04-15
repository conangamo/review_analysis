"""Prepare dataset state for UI demo readiness."""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import get_env
from src.database.db_manager import DatabaseManager
from src.database.models import AspectSentiment, Category, Product, ProductSummary, Review


def _run_command(command):
    print(f"\n$ {' '.join(command)}")
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def _get_counts(db_manager):
    with db_manager.get_session() as session:
        return {
            "categories": session.query(Category).count(),
            "products": session.query(Product).count(),
            "reviews": session.query(Review).count(),
            "aspects": session.query(AspectSentiment).count(),
            "summaries": session.query(ProductSummary).count(),
        }


def main():
    parser = argparse.ArgumentParser(description="Prepare project state for demo-ready UI.")
    parser.add_argument("--category", default="electronics", help="Category for analysis/summaries")
    parser.add_argument("--analysis-limit", type=int, default=1000, help="Limit for auto analysis fix")
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Automatically run missing analysis/summaries steps",
    )
    args = parser.parse_args()

    env = get_env()
    db_path = str(env.get_path("DB_PATH", "./data/processed/reviews.db"))
    db_manager = DatabaseManager(db_path)

    counts = _get_counts(db_manager)
    print("\n=== Demo Readiness Check ===")
    for key, value in counts.items():
        print(f"{key:10s}: {value:,}")

    if counts["products"] == 0 or counts["reviews"] == 0:
        print("\nDatabase is missing core data.")
        print(f"Run: python scripts/parse_data.py --category {args.category}")
        return 1

    if counts["aspects"] == 0:
        if not args.auto_fix:
            print("\nMissing AI aspect results.")
            print(
                f"Run: python scripts/run_analysis.py --category {args.category} --limit {args.analysis_limit}"
            )
            return 1
        _run_command(
            [
                sys.executable,
                "scripts/run_analysis.py",
                "--category",
                args.category,
                "--limit",
                str(args.analysis_limit),
                "--checkpoint-name",
                f"{args.category}_demo_ready",
            ]
        )

    counts = _get_counts(db_manager)
    if counts["summaries"] == 0:
        if not args.auto_fix:
            print("\nMissing product summaries.")
            print(f"Run: python scripts/generate_summaries.py --category {args.category}")
            return 1
        _run_command([sys.executable, "scripts/generate_summaries.py", "--category", args.category])

    final_counts = _get_counts(db_manager)
    print("\n=== Final Demo State ===")
    for key, value in final_counts.items():
        print(f"{key:10s}: {value:,}")
    print("\nDemo ready: open UI now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

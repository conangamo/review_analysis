"""E-comOracle Demo UI v2 (polished, mock-data first)."""

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = Path("data/processed/reviews.db")


def load_mock_data():
    """Mock data for immediate UI visualization."""
    # NOTE: Replace this section with real retrieval + ABSA inference output.
    product_summary = {
        "asin": "B09XS7JWHH",
        "product_name": "Sony WH-1000XM5",
        "total_reviews": 12847,
        "oracle_score": 4.5,
        "category_benchmark": 4.2,
    }

    aspect_scores = pd.DataFrame(
        {
            "aspect": [
                "Battery",
                "Sound Quality",
                "Durability",
                "Comfort",
                "Noise Cancellation",
                "Connectivity",
            ],
            "score": [4.7, 4.8, 3.6, 4.4, 4.9, 3.8],
        }
    )

    sentiment_mix = pd.DataFrame(
        {
            "aspect": [
                "Battery",
                "Sound Quality",
                "Durability",
                "Comfort",
                "Noise Cancellation",
                "Connectivity",
            ],
            "positive": [78, 82, 46, 74, 89, 58],
            "neutral": [14, 10, 21, 16, 7, 19],
            "negative": [8, 8, 33, 10, 4, 23],
        }
    )

    review_snippets = {
        "Battery": [
            "The **battery life** is *amazing* and lasts me several days.",
            "I can use it for long flights; **battery** performance is *excellent*.",
            "Charging speed is quick and practical for daily use.",
        ],
        "Sound Quality": [
            "The **sound quality** is *crisp* with deep bass.",
            "Audio detail is incredible, **sound** feels *premium*.",
            "Vocals are clear even at lower volume.",
        ],
        "Durability": [
            "Build is good, but hinges feel a bit *fragile* after a month.",
            "I expected better **durability** for this price point.",
            "After light drops, scratches appeared on the frame.",
        ],
        "Comfort": [
            "Very light and **comfortable** for long listening sessions.",
            "Ear pads are soft, but clamp pressure can feel *tight* sometimes.",
            "Good fit overall, but gets warm after many hours.",
        ],
        "Noise Cancellation": [
            "**Noise cancellation** is *best-in-class* in noisy offices.",
            "It blocks subway noise surprisingly well.",
            "ANC is strong enough for flights and cafes.",
        ],
        "Connectivity": [
            "Bluetooth pairing is quick, but occasional *dropouts* happen.",
            "Connection is stable most of the time.",
            "Multipoint works well but can switch slowly.",
        ],
    }
    return product_summary, aspect_scores, sentiment_mix, review_snippets


@st.cache_data(ttl=120)
def load_real_products(limit: int = 300) -> pd.DataFrame:
    """Load candidate products from real database."""
    if not DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        query = """
        SELECT
            p.parent_asin AS asin,
            p.title AS product_name,
            COALESCE(ps.total_reviews, COUNT(DISTINCT r.id)) AS total_reviews,
            p.average_rating AS oracle_score
        FROM products p
        LEFT JOIN product_summaries ps ON ps.parent_asin = p.parent_asin
        LEFT JOIN reviews r ON r.parent_asin = p.parent_asin
        WHERE p.is_selected = 1
        GROUP BY p.parent_asin, p.title, ps.total_reviews, p.average_rating
        ORDER BY total_reviews DESC
        LIMIT ?
        """
        return pd.read_sql_query(query, conn, params=(limit,))


def load_real_bundle(selected_asin: str):
    """Load real product summary, aspect scores, mix, and snippets."""
    with sqlite3.connect(DB_PATH) as conn:
        product_row = pd.read_sql_query(
            """
            SELECT
                p.parent_asin AS asin,
                p.title AS product_name,
                COALESCE(ps.total_reviews, COUNT(DISTINCT r.id)) AS total_reviews,
                p.average_rating AS oracle_score,
                (SELECT AVG(average_rating) FROM products WHERE average_rating IS NOT NULL) AS category_benchmark
            FROM products p
            LEFT JOIN product_summaries ps ON ps.parent_asin = p.parent_asin
            LEFT JOIN reviews r ON r.parent_asin = p.parent_asin
            WHERE p.parent_asin = ?
            GROUP BY p.parent_asin, p.title, ps.total_reviews, p.average_rating
            """,
            conn,
            params=(selected_asin,),
        )
        if product_row.empty:
            return None

        sentiment_counts = pd.read_sql_query(
            """
            SELECT
                a.aspect_name AS aspect,
                a.sentiment,
                COUNT(*) AS cnt
            FROM aspect_sentiments a
            JOIN reviews r ON r.id = a.review_id
            WHERE r.parent_asin = ?
            GROUP BY a.aspect_name, a.sentiment
            """,
            conn,
            params=(selected_asin,),
        )

        if sentiment_counts.empty:
            return None

        pivot = (
            sentiment_counts.pivot_table(
                index="aspect",
                columns="sentiment",
                values="cnt",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        for col in ["positive", "neutral", "negative"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot["total"] = pivot["positive"] + pivot["neutral"] + pivot["negative"]
        pivot["score"] = (
            (pivot["positive"] + 0.5 * pivot["neutral"]) / pivot["total"] * 5.0
        ).round(2)
        aspect_scores = pivot[["aspect", "score"]].sort_values("score", ascending=False).reset_index(drop=True)
        sentiment_mix = (
            pivot[["aspect", "positive", "neutral", "negative"]]
            .assign(
                positive=lambda df: (df["positive"] / (df["positive"] + df["neutral"] + df["negative"]) * 100).round(1),
                neutral=lambda df: (df["neutral"] / (df["positive"] + df["neutral"] + df["negative"]) * 100).round(1),
                negative=lambda df: (df["negative"] / (df["positive"] + df["neutral"] + df["negative"]) * 100).round(1),
            )
        )

        snippet_rows = pd.read_sql_query(
            """
            SELECT
                a.aspect_name AS aspect,
                r.text
            FROM aspect_sentiments a
            JOIN reviews r ON r.id = a.review_id
            WHERE r.parent_asin = ?
            ORDER BY a.confidence_score DESC
            LIMIT 300
            """,
            conn,
            params=(selected_asin,),
        )
        review_snippets = {}
        for _, row in snippet_rows.iterrows():
            aspect = row["aspect"]
            if aspect not in review_snippets:
                review_snippets[aspect] = []
            if isinstance(row["text"], str) and len(review_snippets[aspect]) < 3:
                review_snippets[aspect].append(row["text"])

        product_summary = product_row.iloc[0].to_dict()
        if product_summary.get("oracle_score") is None:
            product_summary["oracle_score"] = float(aspect_scores["score"].mean())
        return product_summary, aspect_scores, sentiment_mix, review_snippets


@st.cache_data(ttl=120)
def get_real_snippet_count(selected_asin: str, selected_aspect: str) -> int:
    """Count full number of review snippets for an aspect."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM aspect_sentiments a
            JOIN reviews r ON r.id = a.review_id
            WHERE r.parent_asin = ? AND a.aspect_name = ? AND r.text IS NOT NULL
            """,
            (selected_asin, selected_aspect),
        ).fetchone()
    return int(row[0]) if row else 0


@st.cache_data(ttl=120)
def get_real_snippets_page(
    selected_asin: str,
    selected_aspect: str,
    page: int,
    page_size: int,
) -> list[str]:
    """Load one page of full snippets for selected aspect."""
    offset = max(page - 1, 0) * page_size
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT r.text
            FROM aspect_sentiments a
            JOIN reviews r ON r.id = a.review_id
            WHERE r.parent_asin = ? AND a.aspect_name = ? AND r.text IS NOT NULL
            ORDER BY a.confidence_score DESC, r.helpful_vote DESC, r.id DESC
            LIMIT ? OFFSET ?
            """,
            (selected_asin, selected_aspect, page_size, offset),
        ).fetchall()
    return [r[0] for r in rows if isinstance(r[0], str)]


def score_color_hex(score: float) -> str:
    """Return traffic-light color for score text."""
    if score >= 4.2:
        return "#2E8B57"
    if score >= 3.8:
        return "#F39C12"
    return "#C0392B"


def build_summary_sentence(top_df: pd.DataFrame, low_df: pd.DataFrame) -> str:
    top_part = ", ".join(top_df["aspect"].tolist())
    low_part = ", ".join(low_df["aspect"].tolist())
    return (
        f"Strong performance in {top_part}. "
        f"Main improvement opportunities: {low_part}."
    )


def preview_text(text: str, limit: int = 220) -> tuple[str, bool]:
    """Return preview text and whether it was truncated."""
    if not isinstance(text, str):
        return "", False
    clean = text.strip()
    if len(clean) <= limit:
        return clean, False
    return clean[:limit].rstrip() + "...", True


def render_aspect_scorecards(aspect_scores: pd.DataFrame):
    st.markdown("### Aspect Scorecard")
    st.caption("Core ABSA view - sentiment quality by product aspect")
    aspect_sorted = aspect_scores.sort_values("score", ascending=False).reset_index(drop=True)
    for _, row in aspect_sorted.iterrows():
        aspect = row["aspect"]
        score = float(row["score"])
        color = score_color_hex(score)
        progress_value = int((score / 5.0) * 100)

        label_col, score_col = st.columns([4, 1])
        with label_col:
            st.markdown(f"**{aspect}**")
        with score_col:
            st.markdown(
                f"<div style='text-align:right;color:{color};font-weight:700'>{score:.1f}/5</div>",
                unsafe_allow_html=True,
            )
        st.progress(progress_value)
    return aspect_sorted


def render_analyst_panel(sentiment_mix: pd.DataFrame):
    st.markdown("### Analyst View")
    st.caption("Detailed sentiment composition by aspect")

    chart_df = sentiment_mix.melt(
        id_vars="aspect",
        value_vars=["positive", "neutral", "negative"],
        var_name="sentiment",
        value_name="percent",
    )
    fig = px.bar(
        chart_df,
        x="aspect",
        y="percent",
        color="sentiment",
        barmode="stack",
        title="Aspect Sentiment Composition (%)",
        color_discrete_map={
            "positive": "#2E8B57",
            "neutral": "#95A5A6",
            "negative": "#C0392B",
        },
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sentiment_mix, hide_index=True, use_container_width=True)


def main():
    st.set_page_config(
        page_title="E-comOracle Demo UI v2",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("🔮 E-comOracle")
    st.caption("AI-Driven Predictive Intelligence Platform (ABSA-powered)")

    mode = st.segmented_control(
        "Audience Mode",
        options=["Simple", "Analyst"],
        default="Simple",
        help="Simple for business users, Analyst for expert review.",
    )

    use_real_data = st.toggle(
        "Use real database data",
        value=True,
        help="If disabled (or DB unavailable), app falls back to mock data.",
    )

    real_products = load_real_products() if use_real_data else pd.DataFrame()
    selected_asin = None
    if use_real_data and not real_products.empty:
        search_df = real_products.copy()
        keyword = st.text_input("Search ASIN/Product (real data)", placeholder="Type ASIN or product name")
        if keyword.strip():
            key = keyword.strip().lower()
            search_df = search_df[
                search_df["asin"].str.lower().str.contains(key)
                | search_df["product_name"].str.lower().str.contains(key)
            ]
        if search_df.empty:
            st.warning("No product matches keyword in real data.")
            return
        selected_asin = st.selectbox(
            "Select product from real data",
            options=search_df["asin"].tolist(),
            format_func=lambda asin: f"{asin} | {search_df[search_df['asin'] == asin].iloc[0]['product_name'][:70]}",
        )
        data_bundle = load_real_bundle(selected_asin)
        if data_bundle is None:
            st.warning("Selected product has no ABSA data. Falling back to mock.")
            data_bundle = load_mock_data()
            use_real_data = False
    else:
        data_bundle = load_mock_data()
        if use_real_data:
            st.info("Real DB not found/ready. Using mock data.")

    product_summary, aspect_scores, sentiment_mix, review_snippets = data_bundle

    with st.container(border=True):
        st.subheader("Search Product")
        query = st.text_input(
            "Amazon ASIN or Product Name",
            placeholder="Try: B09XS7JWHH or Sony WH-1000XM5",
        )

        # NOTE: Keep this mock search for offline demo mode.
        matched = (
            query.strip().lower() in product_summary["asin"].lower()
            or query.strip().lower() in product_summary["product_name"].lower()
            or query.strip() == ""
        )

        if matched:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Product", product_summary["product_name"])
            c2.metric("Total Reviews", f"{product_summary['total_reviews']:,}")
            c3.metric("Oracle Score", f"{product_summary['oracle_score']:.1f}/5")
            delta = product_summary["oracle_score"] - product_summary["category_benchmark"]
            c4.metric("Vs Category", f"{product_summary['category_benchmark']:.1f}/5", delta=f"{delta:+.1f}")
        else:
            st.warning("No product found in demo mock data. Try sample ASIN or product name.")
            return

    aspect_sorted = render_aspect_scorecards(aspect_scores)
    top_strengths = aspect_sorted.head(3)
    main_concerns = aspect_sorted.tail(3).sort_values("score", ascending=True)

    st.markdown("### AI Executive Insight")
    st.info(build_summary_sentence(top_strengths, main_concerns))

    st.markdown("### AI Pros & Cons")
    left, right = st.columns(2)
    with left:
        st.markdown("#### 👍 Top Strengths")
        for _, row in top_strengths.iterrows():
            st.success(f"{row['aspect']}: {row['score']:.1f}/5")
    with right:
        st.markdown("#### 👎 Main Concerns")
        for _, row in main_concerns.iterrows():
            st.error(f"{row['aspect']}: {row['score']:.1f}/5")

    if mode == "Analyst":
        render_analyst_panel(sentiment_mix)

    st.markdown("### Customer Snippets")
    selected_aspect = st.selectbox(
        "Choose an aspect to inspect raw review evidence",
        aspect_sorted["aspect"].tolist(),
    )

    page_size = st.selectbox("Snippets per page", options=[5, 10, 20], index=1)
    page_key = f"snippet_page::{selected_aspect}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    with st.expander("Show Raw Review Evidence", expanded=True):
        if use_real_data and selected_asin:
            total_count = get_real_snippet_count(selected_asin, selected_aspect)
            if total_count == 0:
                st.info("No snippets available for this aspect in real data.")
            else:
                total_pages = (total_count + page_size - 1) // page_size
                st.session_state[page_key] = max(1, min(st.session_state[page_key], total_pages))
                current_page = st.session_state[page_key]
                snippets = get_real_snippets_page(
                    selected_asin=selected_asin,
                    selected_aspect=selected_aspect,
                    page=current_page,
                    page_size=page_size,
                )
                st.caption(f"Showing page {current_page}/{total_pages} | total snippets: {total_count}")
                for i, text in enumerate(snippets, start=1 + (current_page - 1) * page_size):
                    preview, is_truncated = preview_text(text)
                    st.markdown(f"**Snippet {i}:** {preview}")
                    if is_truncated:
                        with st.expander(f"View full snippet {i}", expanded=False):
                            st.markdown(text)

                prev_col, mid_col, next_col = st.columns([1, 2, 1])
                with prev_col:
                    if st.button("Previous", disabled=current_page <= 1, key=f"prev::{selected_aspect}"):
                        st.session_state[page_key] = current_page - 1
                        st.rerun()
                with mid_col:
                    goto_page = st.number_input(
                        "Go to page",
                        min_value=1,
                        max_value=total_pages,
                        value=current_page,
                        step=1,
                        key=f"goto::{selected_aspect}",
                    )
                    if int(goto_page) != current_page:
                        st.session_state[page_key] = int(goto_page)
                        st.rerun()
                with next_col:
                    if st.button("Next", disabled=current_page >= total_pages, key=f"next::{selected_aspect}"):
                        st.session_state[page_key] = current_page + 1
                        st.rerun()
        else:
            snippets = review_snippets.get(selected_aspect, [])
            if not snippets:
                st.info("No snippets available for this aspect in mock data.")
            for i, text in enumerate(snippets[:3], start=1):
                preview, is_truncated = preview_text(text)
                st.markdown(f"**Snippet {i}:** {preview}")
                if is_truncated:
                    with st.expander(f"View full snippet {i}", expanded=False):
                        st.markdown(text)

    st.markdown("---")
    st.caption(
        "Integration note: replace `load_mock_data()` and the simple search matching with your real "
        "ABSA pipeline + product retrieval service."
    )


if __name__ == "__main__":
    main()

"""
Thử nghiệm bình luận tự do — ưu tiên logic model (ZeroShotClassifier), tùy chọn pipeline SentimentAnalyzer.
Chạy: streamlit run src/ui/app.py → chọn trang trong sidebar.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd

from src.core import ConfigLoader, AspectManager, get_env
from src.ai_engine import ZeroShotClassifier, SentimentAnalyzer
from src.ui.components.review_cards import render_review_card
from src.ui.utils.formatters import format_aspect_name


def _resolve_analysis_settings(config_loader, category_config, env):
    """Khớp scripts/run_analysis.py::_resolve_analysis_settings."""
    model_config = config_loader.load_model_config()
    analyzer_config = model_config.get("sentiment_analyzer", {})
    optimization_config = model_config.get("optimization", {})
    category_processing = category_config.get("processing", {})
    cache_dir = env.get_path("CACHE_DIR", "./data/cache")
    checkpoint_dir = env.get_path("CHECKPOINT_DIR", str(cache_dir / "checkpoints"))
    return {
        "use_keyword_filter": env.get_bool(
            "USE_KEYWORD_FILTER", optimization_config.get("use_keyword_filter", True)
        ),
        "confidence_threshold": env.get_float(
            "CONFIDENCE_THRESHOLD", analyzer_config.get("confidence_threshold", 0.65)
        ),
        "min_confidence_tier1": env.get_float(
            "MIN_CONFIDENCE_TIER1", analyzer_config.get("min_confidence_tier1", 0.55)
        ),
        "min_confidence_tier2": env.get_float(
            "MIN_CONFIDENCE_TIER2", analyzer_config.get("min_confidence_tier2", 0.70)
        ),
        "min_confidence_neutral": env.get_float(
            "MIN_CONFIDENCE_NEUTRAL", analyzer_config.get("min_confidence_neutral", 0.40)
        ),
        "analysis_batch_size": env.get_int(
            "ANALYSIS_BATCH_SIZE", category_processing.get("ai_batch_size", 32)
        ),
        "checkpoint_dir": checkpoint_dir,
    }


def _category_slugs():
    d = _ROOT / "config" / "categories"
    if not d.is_dir():
        return ["electronics"]
    slugs = sorted(
        p.stem for p in d.glob("*.yaml") if p.stem != "_template" and p.is_file()
    )
    return slugs or ["electronics"]


@st.cache_resource(show_spinner="Đang tải mô hình (lần đầu có thể vài phút)...")
def _get_playground_bundle(category_slug: str) -> Dict[str, Any]:
    """Một classifier + aspect_manager / settings — dùng chung cho logic model và SentimentAnalyzer."""
    env = get_env()
    config_loader = ConfigLoader()
    category_config = config_loader.load_category_config(category_slug)
    settings = _resolve_analysis_settings(config_loader, category_config, env)

    model_config = config_loader.load_model_config()
    zs = model_config.get("zero_shot", {})
    model_name = env.get_str("MODEL_NAME", zs.get("model_name", "valhalla/distilbart-mnli-12-3"))
    use_gpu = env.get_bool("USE_GPU", True)
    use_fp16 = env.get_bool("USE_FP16", True)
    batch_size = env.get_int("BATCH_SIZE", zs.get("batch_size", 32))
    device = "cuda" if use_gpu else "cpu"

    aspect_manager = AspectManager(category_slug, config_loader)
    classifier = ZeroShotClassifier(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
        batch_size=batch_size,
        max_length=int(zs.get("max_length", 512)),
    )
    return {
        "classifier": classifier,
        "aspect_manager": aspect_manager,
        "settings": settings,
        "env": env,
        "category_slug": category_slug,
    }


def _run_zero_shot_per_aspect(
    text: str,
    bundle: Dict[str, Any],
    *,
    all_aspects: bool = False,
) -> List[Dict[str, Any]]:
    """
    Logic model: ZeroShotClassifier.classify_sentiment cho từng aspect được chọn.

    Mặc định chỉ gọi model cho aspect **có keyword trong câu** (cùng ý tưởng
    `AspectManager.get_aspects_for_analysis` / bước 1 của SentimentAnalyzer).
    Gọi `classify_sentiment` cho *mọi* aspect trong YAML khiến NLI thường gán
    polar giống nhau theo âm điệu chung — trông như “bịa” aspect.

    `all_aspects=True`: lặp toàn bộ YAML (chỉ dùng khi debug / so sánh nhiễu).
    """
    clf: ZeroShotClassifier = bundle["classifier"]
    am: AspectManager = bundle["aspect_manager"]
    if all_aspects:
        to_score = am.aspects
    else:
        to_score = am.get_aspects_for_analysis(
            text,
            include_tier1_always=False,
            strict_keyword_matching=True,
        )
    rows = []
    for asp in to_score:
        name = asp["name"]
        r = clf.classify_sentiment(text, name)
        rows.append(
            {
                "aspect": name,
                "tier": asp.get("tier"),
                "sentiment": r["sentiment"],
                "confidence": float(r.get("confidence", 0) or 0),
                "all_scores": r.get("all_scores", {}),
                "detection_method": "zero_shot",
            }
        )
    return rows


def _run_zero_shot_nli_presence(
    text: str,
    bundle: Dict[str, Any],
    *,
    presence_min: float = 0.30,
    max_aspects: int = 8,
) -> List[Dict[str, Any]]:
    """
    Chọn aspect **chỉ bằng model**: `classify_aspect_presence` (multi-label NLI
    *This review mentions {aspect}.*), lọc theo `presence_min`, giới hạn `max_aspects`,
    rồi `classify_sentiment` — **không** dùng tier/keyword/min_mentions từ YAML.

    YAML vẫn cung cấp **danh sách tên aspect** (từ vựng nhãn); không có danh sách
    thì zero-shot không biết cần hỏi những khía cạnh nào.
    """
    clf: ZeroShotClassifier = bundle["classifier"]
    am: AspectManager = bundle["aspect_manager"]
    names = [a["name"] for a in am.aspects]
    if not names:
        return []
    presence = clf.classify_aspect_presence(text, names)
    ranked: List[Tuple[str, float]] = sorted(
        presence.items(), key=lambda x: x[1], reverse=True
    )
    selected = [(n, float(s)) for n, s in ranked if s >= presence_min][: max(1, max_aspects)]
    rows: List[Dict[str, Any]] = []
    for name, p_score in selected:
        r = clf.classify_sentiment(text, name)
        rows.append(
            {
                "aspect": name,
                "tier": None,
                "sentiment": r["sentiment"],
                "confidence": float(r.get("confidence", 0) or 0),
                "all_scores": r.get("all_scores", {}),
                "detection_method": "nli_mentions_then_sentiment",
                "presence_score": p_score,
            }
        )
    return rows


def _run_stable_hybrid(
    text: str,
    bundle: Dict[str, Any],
    *,
    nli_extra_min: float = 0.42,
    max_nli_extras: int = 3,
    max_total_aspects: int = 10,
) -> List[Dict[str, Any]]:
    """
    Cách **ổn định** trên comment tự do (trade-off thực tế):

    1. Lấy aspect khớp **keyword YAML** (độ chính xác cao, có tier).
    2. Chạy NLI *mentions* một lần; **bổ sung** tối đa `max_nli_extras` aspect chưa có
       trong (1) nếu P(mentions) ≥ `nli_extra_min` (ngưỡng **cao** để hạn chế bịa).
    3. Giới hạn tổng số aspect ≤ `max_total_aspects`, rồi `classify_sentiment`.

    Không thay thế pipeline DB; đây là preset hợp lý khi người dùng dán câu bất kỳ.
    """
    clf: ZeroShotClassifier = bundle["classifier"]
    am: AspectManager = bundle["aspect_manager"]
    kw_list = am.get_aspects_for_analysis(
        text,
        include_tier1_always=False,
        strict_keyword_matching=True,
    )
    by_name: Dict[str, Dict[str, Any]] = {a["name"]: dict(a) for a in kw_list}

    names = [a["name"] for a in am.aspects]
    presence: Dict[str, float] = {}
    if names:
        presence = clf.classify_aspect_presence(text, names)
        ranked = sorted(presence.items(), key=lambda x: x[1], reverse=True)
        extras = 0
        for n, sc in ranked:
            if len(by_name) >= max_total_aspects:
                break
            if n in by_name:
                continue
            if float(sc) < nli_extra_min:
                continue
            if extras >= max_nli_extras:
                break
            asp = am.get_aspect(n)
            if asp is not None:
                by_name[n] = dict(asp)
                extras += 1

    order: List[str] = [a["name"] for a in kw_list]
    for n in by_name:
        if n not in order:
            order.append(n)

    rows: List[Dict[str, Any]] = []
    for name in order:
        asp = by_name[name]
        r = clf.classify_sentiment(text, name)
        psc = float(presence[name]) if name in presence else None
        rows.append(
            {
                "aspect": name,
                "tier": asp.get("tier"),
                "sentiment": r["sentiment"],
                "confidence": float(r.get("confidence", 0) or 0),
                "all_scores": r.get("all_scores", {}),
                "detection_method": "hybrid_keyword_then_nli_extra",
                "presence_score": psc,
            }
        )
    return rows


def _overall_zero_shot(text: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Overall: cùng cách SentimentAnalyzer.get_overall_sentiment — chỉ gọi classifier."""
    clf: ZeroShotClassifier = bundle["classifier"]
    result = clf.classify(
        text,
        ["positive", "negative", "neutral"],
        multi_label=False,
        hypothesis_template="This review is {}.",
    )
    return {
        "sentiment": result["labels"][0],
        "confidence": result["scores"][0],
        "all_scores": {lab: sc for lab, sc in zip(result["labels"], result["scores"])},
    }


def _build_sentiment_analyzer(bundle: Dict[str, Any], free_text_mode: bool) -> SentimentAnalyzer:
    """Cùng classifier đã cache; thêm lớp SentimentAnalyzer (negation, heuristic, ngưỡng tier…)."""
    settings = bundle["settings"]
    slug = bundle["category_slug"]
    if free_text_mode:
        use_kf = False
        require_ev = False
    else:
        use_kf = settings["use_keyword_filter"]
        require_ev = True

    return SentimentAnalyzer(
        category_name=slug,
        aspect_manager=bundle["aspect_manager"],
        classifier=bundle["classifier"],
        use_keyword_filter=use_kf,
        confidence_threshold=settings["confidence_threshold"],
        min_confidence_tier1=settings["min_confidence_tier1"],
        min_confidence_tier2=settings["min_confidence_tier2"],
        min_confidence_neutral=settings["min_confidence_neutral"],
        require_keyword_evidence=require_ev,
    )


st.set_page_config(
    page_title="Playground — Product Review Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    with st.sidebar:
        st.markdown("### Playground")
        if st.button("Xóa cache (model + analyzer)", key="pg_clear_cache"):
            st.cache_resource.clear()
            st.rerun()

    st.title("🧪 Thử bình luận tự do")

    slugs = _category_slugs()
    category = st.selectbox("Category", options=slugs, index=0, key="pg_cat")

    default_text = (
        "The battery lasts all day which is great. Screen is a bit dim outdoors. "
        "Overall good value for the price."
    )
    text = st.text_area("Bình luận (tiếng Anh)", value=default_text, height=160, key="pg_text")

    aspect_mode = st.radio(
        "Cách chọn aspect",
        options=("hybrid_stable", "keyword_yaml", "nli_presence", "all_yaml"),
        index=0,
        format_func=lambda v: {
            "hybrid_stable": "Ổn định (khuyến nghị)",
            "keyword_yaml": "Keyword YAML",
            "nli_presence": "NLI mentions",
            "all_yaml": "Mọi aspect YAML",
        }[v],
        key="pg_aspect_mode",
        horizontal=True,
    )
    presence_min = 0.30
    max_aspects_nli = 8
    if aspect_mode == "nli_presence":
        cpm, cmx = st.columns(2)
        with cpm:
            presence_min = st.slider(
                "P(mentions) tối thiểu",
                min_value=0.05,
                max_value=0.55,
                value=0.30,
                step=0.02,
                key="pg_pres_min",
            )
        with cmx:
            max_aspects_nli = st.slider(
                "Tối đa aspect",
                min_value=2,
                max_value=14,
                value=8,
                step=1,
                key="pg_max_asp",
            )

    show_pipeline = st.checkbox(
        "Thêm SentimentAnalyzer",
        value=False,
        key="pg_show_sa",
    )
    strict_like_db = st.checkbox(
        "Giống lúc lưu DB",
        value=False,
        disabled=not show_pipeline,
        key="pg_strict",
    )

    run = st.button("Phân tích", type="primary", key="pg_run")

    if run:
        if not (text or "").strip():
            st.warning("Nhập nội dung bình luận.")
        else:
            t = text.strip()
            with st.spinner("Đang chạy model..."):
                bundle = _get_playground_bundle(category)
                if aspect_mode == "hybrid_stable":
                    model_rows = _run_stable_hybrid(t, bundle)
                elif aspect_mode == "nli_presence":
                    model_rows = _run_zero_shot_nli_presence(
                        t,
                        bundle,
                        presence_min=float(presence_min),
                        max_aspects=int(max_aspects_nli),
                    )
                elif aspect_mode == "all_yaml":
                    model_rows = _run_zero_shot_per_aspect(t, bundle, all_aspects=True)
                else:
                    model_rows = _run_zero_shot_per_aspect(t, bundle, all_aspects=False)
                model_overall = _overall_zero_shot(t, bundle)

                sa_rows = None
                sa_overall = None
                if show_pipeline:
                    analyzer = _build_sentiment_analyzer(bundle, free_text_mode=not strict_like_db)
                    sa_rows = analyzer.analyze_review(t, review_id=None)
                    sa_overall = analyzer.get_overall_sentiment(t)

            # Không gán vào key trùng widget (pg_text, pg_cat, …) — Streamlit sẽ lỗi sau khi widget đã render.
            st.session_state["pg_snap_text"] = t
            st.session_state["pg_snap_cat"] = category
            st.session_state["pg_model_rows"] = model_rows
            st.session_state["pg_model_overall"] = model_overall
            st.session_state["pg_sa_rows"] = sa_rows
            st.session_state["pg_sa_overall"] = sa_overall
            st.session_state["pg_snap_strict"] = strict_like_db
            st.session_state["pg_snap_aspect_mode"] = aspect_mode
            st.session_state["pg_snap_presence_min"] = float(presence_min)
            st.session_state["pg_snap_max_aspects_nli"] = int(max_aspects_nli)

    if st.session_state.get("pg_model_rows") is None:
        return

    if (
        st.session_state.get("pg_snap_cat") != category
        or st.session_state.get("pg_snap_aspect_mode") != aspect_mode
    ):
        st.warning("Cài đặt đã đổi — bấm **Phân tích** lại.")

    t = st.session_state["pg_snap_text"]
    model_rows: List[Dict[str, Any]] = st.session_state["pg_model_rows"]
    model_overall: Dict[str, Any] = st.session_state["pg_model_overall"]

    st.markdown("---")
    st.subheader("Kết quả model")

    if not model_rows:
        st.info("Không có aspect nào — thử chế độ khác hoặc chỉnh ngưỡng NLI.")

    mo = model_overall or {}
    st.metric(
        "Overall",
        str(mo.get("sentiment", "—")).title(),
        delta=f"{float(mo.get('confidence', 0) or 0) * 100:.0f}%",
    )

    aspects_card = [
        {
            "aspect_name": r["aspect"],
            "sentiment": r["sentiment"],
            "confidence_score": r["confidence"],
            "tier": r.get("tier"),
            "negation_adjusted": False,
        }
        for r in model_rows
    ]
    review_stub = {
        "id": 0,
        "rating": 4.0,
        "title": "Playground — output model",
        "text": t,
        "timestamp": None,
        "verified_purchase": False,
        "helpful_vote": 0,
    }
    render_review_card(review_stub, aspects_card, show_confidence=True, show_validation=False)

    if model_rows:
        base_cols = {
            "Aspect": [format_aspect_name(r["aspect"]) for r in model_rows],
            "Sentiment": [r["sentiment"] for r in model_rows],
            "Confidence %": [round(r["confidence"] * 100, 1) for r in model_rows],
            "Tier": [r.get("tier") for r in model_rows],
        }
        if any(r.get("presence_score") is not None for r in model_rows):
            base_cols["P(mentions) %"] = [
                round(float(r["presence_score"]) * 100, 1) if r.get("presence_score") is not None else None
                for r in model_rows
            ]
        st.dataframe(pd.DataFrame(base_cols), use_container_width=True, hide_index=True)

    if st.session_state.get("pg_sa_rows") is not None:
        st.markdown("---")
        st.subheader("SentimentAnalyzer")
        so = st.session_state.get("pg_sa_overall") or {}
        st.metric(
            "Overall",
            str(so.get("sentiment", "—")).title(),
            delta=f"{float(so.get('confidence', 0) or 0) * 100:.0f}%",
        )
        sa_card = [
            {
                "aspect_name": r.get("aspect", ""),
                "sentiment": r.get("sentiment", "neutral"),
                "confidence_score": float(r.get("confidence", 0) or 0),
                "tier": r.get("tier"),
                "negation_adjusted": r.get("negation_adjusted", False),
            }
            for r in st.session_state["pg_sa_rows"]
        ]
        review_stub2 = dict(review_stub)
        review_stub2["title"] = "Sau SentimentAnalyzer"
        render_review_card(review_stub2, sa_card, show_confidence=True, show_validation=True)


main()

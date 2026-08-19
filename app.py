"""
app.py — Marathi Sentiment Analysis Web Demo
=============================================
Run with:  streamlit run app.py
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Marathi Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark premium look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1c2128, #21262d);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(88,166,255,0.15);
}
.metric-value { font-size: 2rem; font-weight: 700; color: #58a6ff; }
.metric-label { font-size: 0.8rem; color: #8b949e; margin-top: 0.2rem; letter-spacing: 0.05em; text-transform: uppercase; }

/* Prediction result box */
.pred-box {
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    border: 1px solid;
    text-align: center;
}
.pred-positive { background: linear-gradient(135deg,#0d2b1d,#1a3a28); border-color: #238636; }
.pred-negative { background: linear-gradient(135deg,#2b0d0d,#3a1a1a); border-color: #da3633; }
.pred-neutral  { background: linear-gradient(135deg,#1a1a2b,#212140); border-color: #6e40c9; }
.pred-title    { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem; }
.pred-sub      { font-size: 0.95rem; color: #8b949e; }

/* Confidence bar container */
.conf-bar-bg {
    background: #21262d;
    border-radius: 999px;
    height: 10px;
    margin: 4px 0 12px 0;
    overflow: hidden;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #21262d;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: #21262d;
    border-radius: 8px 8px 0 0;
    color: #8b949e;
    padding: 0.5rem 1.2rem;
    border: 1px solid #30363d;
    border-bottom: none;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1c2128;
    color: #58a6ff;
    border-color: #58a6ff;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #388bfd, #58a6ff);
    box-shadow: 0 4px 15px rgba(56,139,253,0.4);
    transform: translateY(-1px);
}

/* Text area */
.stTextArea textarea {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    color: #e6edf3;
    font-size: 1rem;
}

/* Info boxes */
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABEL_COLORS = {
    "positive": ("#3fb950", "pred-positive", "Positive Sentiment ↑"),
    "negative": ("#f85149", "pred-negative", "Negative Sentiment ↓"),
    "neutral":  ("#8957e5", "pred-neutral",  "Neutral Sentiment →"),
}

MODELS = {
    "sarvamai/sarvam-1 (Fine-tuned)": "saved_models/sarvam-1-lora",
    "MuRIL (Google)":                 "outputs/google--muril-base-cased",
    "IndicBERT (ai4bharat)":          "outputs/ai4bharat--indic-bert",
}

MODEL_DISPLAY_NAMES = {
    "sarvam-1": "Sarvam-1 (2B Generative SLM - LoRA)",
    "sarvamai--sarvam-1-lora": "Sarvam-1 (2B Generative SLM - LoRA)",
    "google--muril-base-cased": "MuRIL (Google Multilingual BERT)",
    "ai4bharat--indic-bert": "IndicBERT (AI4Bharat)",
}

def model_sort_order(path_or_str) -> int:
    tag = str(path_or_str).lower()
    if "sarvam" in tag:
        return 0
    if "muril" in tag:
        return 1
    if "indic" in tag:
        return 2
    return 3

RESULTS_DIR   = Path("results")
DOMAIN_DIR    = RESULTS_DIR / "domain_eval"
CODE_MIX_DIR  = RESULTS_DIR / "code_mixed_eval"


def get_available_models() -> dict[str, str]:
    """Return only model dirs that actually exist on disk, ordered Sarvam-1 -> MuRIL -> IndicBERT."""
    available = {}
    for name, path in MODELS.items():
        p = Path(path)
        if p.exists() and ((p / "config.json").exists() or (p / "adapter_config.json").exists()):
            available[name] = path
    return available


@st.cache_resource(show_spinner="Loading model…")
def load_predictor(model_dir: str):
    """Cache the predictor so it isn't reloaded on every interaction."""
    from predict import SentimentPredictor
    from train import detect_device
    device = detect_device(force_cpu=True)
    return SentimentPredictor(model_dir=model_dir, device=device)


def make_conf_bars(scores: dict[str, float]) -> None:
    """Render confidence bars for each class using Streamlit progress."""
    for cls, prob in scores.items():
        color, _, _ = LABEL_COLORS.get(cls, ("#8b949e", "", ""))
        pct = prob * 100
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(prob, text=f"**{cls.capitalize()}**")
        with col2:
            st.markdown(
                f"<div style='text-align:right;color:{color};font-weight:600;"
                f"padding-top:6px'>{pct:.1f}%</div>",
                unsafe_allow_html=True,
            )


def render_prediction(result: dict, model_label: str = "") -> None:
    """Display a formatted prediction result card."""
    label  = result["label"]
    conf   = result["confidence"] * 100
    color, css_class, title = LABEL_COLORS.get(label, ("#8b949e", "pred-neutral", label))

    if model_label:
        st.markdown(f"<div class='section-header'>{model_label}</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='pred-box {css_class}'>
        <div class='pred-title' style='color:{color}'>{title}</div>
        <div class='pred-sub'>{conf:.1f}% confidence &nbsp;·&nbsp; {result['latency_ms']:.1f} ms latency</div>
    </div>
    """, unsafe_allow_html=True)

    make_conf_bars(result["scores"])

    if "explanation" in result and result["explanation"]:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg, #161b22, #1c2128); border-left: 4px solid #58a6ff; border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0;'>
            <div style='color: #58a6ff; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.3rem;'>
                💡 Explainable AI (XAI) Reasoning:
            </div>
            <div style='color: #e6edf3; font-size: 0.95rem; line-height: 1.5;'>
                {result['explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 Normalized input & SLM generation details"):
        st.markdown(f"**Normalized text:** `{result['input']}`")
        if "raw_generation" in result:
            st.markdown(f"**Raw SLM Generation:** `{result['raw_generation']}`")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎭 Marathi Sentiment")
    st.markdown("*NLP Project — Fine-tuned Multilingual BERT*")
    st.divider()

    available = get_available_models()
    if not available:
        st.error("No trained models found in `outputs/`.\nRun `python src/train.py --cpu` first.")
        st.stop()

    st.markdown("#### Select Model")
    selected_model_name = st.selectbox(
        "Model",
        options=list(available.keys()),
        label_visibility="collapsed",
    )
    selected_model_dir = available[selected_model_name]

    st.markdown("#### Compare Models")
    compare_mode = st.toggle("Run both side-by-side", value=False)
    if compare_mode and len(available) >= 2:
        compare_model_name = st.selectbox(
            "Second Model",
            options=[k for k in available if k != selected_model_name],
            label_visibility="collapsed",
        )
        compare_model_dir = available[compare_model_name]

    st.divider()
    model_path = Path(selected_model_dir)
    st.markdown(f"**Checkpoint:** `{model_path.name}`")

    # Show model parameters if config exists
    try:
        import json
        if (model_path / "adapter_config.json").exists():
            cfg = json.loads((model_path / "adapter_config.json").read_text())
            r = cfg.get("r", 16)
            alpha = cfg.get("lora_alpha", 32)
            base = cfg.get("base_model_name_or_path", "sarvamai/sarvam-1")
            st.markdown(f"**Base SLM:** `{base}`")
            st.markdown(f"**LoRA Config:** rank `r={r}`, `alpha={alpha}` (NF4)")
        elif (model_path / "config.json").exists():
            cfg = json.loads((model_path / "config.json").read_text())
            hidden = cfg.get("hidden_size", "?")
            layers = cfg.get("num_hidden_layers", "?")
            st.markdown(f"**Architecture:** {layers} layers × {hidden} hidden")
    except Exception:
        pass

    st.divider()
    st.markdown(
        "<div style='font-size:0.75rem;color:#8b949e;'>"
        "Dataset: L3Cube-MahaSent-MD<br>"
        "Labels: Negative · Neutral · Positive<br>"
        "Supports Devanagari + code-mixed text"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem 0;'>
    <h1 style='font-size:2.5rem;font-weight:700;
               background:linear-gradient(90deg,#58a6ff,#8957e5);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               margin-bottom:0.3rem;'>
        Marathi Sentiment Analyzer
    </h1>
    <p style='color:#8b949e;font-size:1rem;'>
        Fine-tuned multilingual BERT models for Marathi, Devanagari &amp; code-mixed text
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Live Predict",
    "📊 Model Results",
    "🌐 Domain Analysis",
    "🔀 Code-Mixed Eval",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Predict
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Type any Marathi sentence below")
    st.markdown(
        "<div style='color:#8b949e;font-size:0.9rem;margin-bottom:1rem;'>"
        "Supports Devanagari (मराठी), Romanized (khup chan aahe!), "
        "and code-mixed text (khup boring movie aahe, waste of time)."
        "</div>",
        unsafe_allow_html=True,
    )

    # Quick example buttons
    examples = [
        "मला हा चित्रपट खूप आवडला!",
        "khup boring movie aahe, waste of time",
        "mala khup raga ala, worst experience ever",
        "thik aahe, na too good na too bad",
        "bahut accha aahe bhai, truly amazing story",
        "he product ekdum bakwaas aahe, don't buy",
    ]

    st.markdown("<div class='section-header'>Quick Examples</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    example_clicked = None
    for i, ex in enumerate(examples):
        with cols[i % 3]:
            if st.button(f'"{ex[:30]}…"' if len(ex) > 30 else f'"{ex}"',
                         key=f"ex_{i}", use_container_width=True):
                example_clicked = ex

    st.markdown("<div class='section-header'>Your Input</div>", unsafe_allow_html=True)
    user_text = st.text_area(
        "Enter text",
        value=example_clicked if example_clicked else "",
        height=100,
        placeholder="Type a Marathi sentence here… e.g. 'मला हे गाणे खूप आवडले!'",
        label_visibility="collapsed",
    )

    c_btn, c_xai = st.columns([2, 2])
    with c_btn:
        analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)
    with c_xai:
        enable_xai = st.checkbox("💡 Generate Explainable AI Reasoning (XAI)", value=True)

    if analyze_btn and user_text.strip():
        with st.spinner("Running inference & reasoning…"):
            predictor1 = load_predictor(selected_model_dir)
            result1    = predictor1.predict(user_text, with_reasoning=enable_xai)

            if compare_mode and len(available) >= 2:
                predictor2 = load_predictor(compare_model_dir)
                result2    = predictor2.predict(user_text, with_reasoning=enable_xai)

        st.divider()

        if compare_mode and len(available) >= 2:
            # Side-by-side
            col_a, col_b = st.columns(2)
            with col_a:
                render_prediction(result1, model_label=f"🔵 {selected_model_name}")
            with col_b:
                render_prediction(result2, model_label=f"🟣 {compare_model_name}")

            # Agreement indicator
            if result1["label"] == result2["label"]:
                st.success(f"✅ Both models agree: **{result1['label'].capitalize()}**")
            else:
                st.warning(
                    f"⚠️ Models disagree — **{selected_model_name}** says "
                    f"*{result1['label']}*, **{compare_model_name}** says *{result2['label']}*"
                )
        else:
            render_prediction(result1, model_label=selected_model_name)

    elif analyze_btn:
        st.warning("Please enter some text first.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Results
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Model Evaluation Results")

    # Scan for any eval summary CSVs
    eval_csvs = list(RESULTS_DIR.glob("*_eval_summary.csv"))
    baseline_csv = RESULTS_DIR / "baseline_metrics.csv"

    if not eval_csvs and not baseline_csv.exists():
        st.info(
            "No evaluation results found yet.\n\n"
            "Run `python src/evaluate.py --cpu --data_dir data` after training to generate results."
        )
    else:
        # Baseline
        if baseline_csv.exists():
            st.markdown("<div class='section-header'>TF-IDF Baseline</div>", unsafe_allow_html=True)
            df_base = pd.read_csv(baseline_csv)
            cols = st.columns(len(df_base.columns))
            for col, (cname, val) in zip(cols, df_base.iloc[0].items()):
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{float(val):.3f}</div>
                        <div class='metric-label'>{cname.replace('_', ' ')}</div>
                    </div>""", unsafe_allow_html=True)

        # Neural model eval CSVs — sorted: Sarvam-1 -> MuRIL -> IndicBERT
        for csv_path in sorted(eval_csvs, key=lambda p: model_sort_order(p.stem)):
            model_tag = csv_path.stem.replace("_eval_summary", "")
            display_title = MODEL_DISPLAY_NAMES.get(model_tag, model_tag)
            st.markdown(f"<div class='section-header'>{display_title}</div>", unsafe_allow_html=True)
            df = pd.read_csv(csv_path)
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            display_cols = st.columns(len(numeric_cols))
            for col, cname in zip(display_cols, numeric_cols):
                val = df[cname].iloc[0]
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{float(val):.3f}</div>
                        <div class='metric-label'>{cname.replace('_', ' ')}</div>
                    </div>""", unsafe_allow_html=True)

            # Per-class metrics
            per_class_csv = RESULTS_DIR / f"{model_tag}_per_class_metrics.csv"
            if per_class_csv.exists():
                with st.expander("Per-class breakdown"):
                    st.dataframe(
                        pd.read_csv(per_class_csv),
                        use_container_width=True,
                        hide_index=True,
                    )

            # Confusion matrix
            cm_png = RESULTS_DIR / f"{model_tag}_confusion_matrix.png"
            if cm_png.exists():
                with st.expander("Confusion Matrix"):
                    st.image(str(cm_png), use_container_width=True)

            # Loss curves
            loss_png = RESULTS_DIR / f"{model_tag}_loss_curves.png"
            if loss_png.exists():
                with st.expander("Training & Validation Loss Curves"):
                    st.image(str(loss_png), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Domain Analysis
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Cross-Domain Performance")
    st.markdown(
        "<div style='color:#8b949e;font-size:0.9rem;margin-bottom:1rem;'>"
        "Evaluates how well the model generalises across four different domains: "
        "Movie Reviews, Generic Tweets, TV Subtitles, and Political Tweets."
        "</div>",
        unsafe_allow_html=True,
    )

    domain_csvs = list(DOMAIN_DIR.glob("*_domain_summary.csv"))

    if not domain_csvs:
        st.info(
            "No domain evaluation results yet.\n\n"
            "Run `python src/domain_eval.py --data_dir data` after training."
        )
    else:
        # Sorted: Sarvam-1 -> MuRIL -> IndicBERT
        for csv_path in sorted(domain_csvs, key=lambda p: model_sort_order(p.stem)):
            model_tag = csv_path.stem.replace("_domain_summary", "")
            display_title = MODEL_DISPLAY_NAMES.get(model_tag, model_tag)
            st.markdown(f"<div class='section-header'>{display_title}</div>", unsafe_allow_html=True)

            df = pd.read_csv(csv_path)
            st.dataframe(
                df.style.format({
                    "macro_f1": "{:.3f}",
                    "macro_precision": "{:.3f}",
                    "macro_recall": "{:.3f}",
                    "ms_per_sample": "{:.1f} ms",
                }),
                use_container_width=True,
                hide_index=True,
            )

            comp_png   = DOMAIN_DIR / f"{model_tag}_domain_comparison.png"
            latency_png = DOMAIN_DIR / f"{model_tag}_latency.png"

            img_col1, img_col2 = st.columns(2)
            if comp_png.exists():
                with img_col1:
                    st.image(str(comp_png), caption="Domain F1 Comparison", use_container_width=True)
            if latency_png.exists():
                with img_col2:
                    st.image(str(latency_png), caption="Inference Latency by Domain", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Code-Mixed Eval
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### Code-Mixed Text Evaluation")
    st.markdown(
        "<div style='color:#8b949e;font-size:0.9rem;margin-bottom:1rem;'>"
        "Evaluates the model on 30 curated Romanized Marathi / Hindi-English "
        "code-mixed social-media sentences (10 per sentiment class)."
        "</div>",
        unsafe_allow_html=True,
    )

    cm_csvs = list(CODE_MIX_DIR.glob("*_code_mixed_results.csv"))

    if not cm_csvs:
        st.info(
            "No code-mixed evaluation results yet.\n\n"
            "Run `python src/code_mixed_eval.py` after training."
        )
    else:
        # Sorted: Sarvam-1 -> MuRIL -> IndicBERT
        for csv_path in sorted(cm_csvs, key=lambda p: model_sort_order(p.stem)):
            model_tag = csv_path.stem.replace("_code_mixed_results", "")
            display_title = MODEL_DISPLAY_NAMES.get(model_tag, model_tag)
            st.markdown(f"<div class='section-header'>{display_title}</div>", unsafe_allow_html=True)

            df = pd.read_csv(csv_path)
            accuracy = df["correct"].mean() * 100
            avg_lat  = df["latency_ms"].mean()

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='metric-card'><div class='metric-value'>{accuracy:.1f}%</div><div class='metric-label'>Accuracy</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><div class='metric-value'>{df['correct'].sum()}/30</div><div class='metric-label'>Correct</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_lat:.1f}ms</div><div class='metric-label'>Avg Latency</div></div>", unsafe_allow_html=True)

            st.markdown("")

            # Colour-code correct/incorrect rows
            def highlight_rows(row):
                color = "background-color:#0d2b1d;" if row["correct"] else "background-color:#2b0d0d;"
                return [color] * len(row)

            with st.expander("View all 30 sentences"):
                st.dataframe(
                    df.style.apply(highlight_rows, axis=1).format({"confidence": "{:.2%}", "latency_ms": "{:.1f} ms"}),
                    use_container_width=True,
                    hide_index=True,
                )

            cm_png = CODE_MIX_DIR / f"{model_tag}_code_mixed_confusion.png"
            if cm_png.exists():
                col_l, col_r = st.columns([1, 1])
                with col_l:
                    st.image(str(cm_png), caption="Code-Mixed Confusion Matrix", use_container_width=True)

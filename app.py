import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import re

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InstructABSA",
    page_icon="🔍",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0d0d;
    color: #f0ece4;
}

.stApp {
    background: #0d0d0d;
}

/* Header */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    background: linear-gradient(135deg, #f0ece4 30%, #c8a97e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #666;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #2a2a2a;
    margin: 1.5rem 0;
}

/* Text area override */
textarea {
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
    color: #f0ece4 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
}
textarea:focus {
    border-color: #c8a97e !important;
    box-shadow: 0 0 0 2px rgba(200,169,126,0.15) !important;
}

/* Button */
.stButton > button {
    background: #c8a97e;
    color: #0d0d0d;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    padding: 0.6rem 2rem;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    background: #e0c49a;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(200,169,126,0.3);
}

/* Result cards */
.result-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: border-color 0.2s;
}
.result-card:hover {
    border-color: #3a3a3a;
}
.aspect-term {
    font-family: 'DM Mono', monospace;
    font-size: 1rem;
    color: #f0ece4;
    font-weight: 500;
}
.sentiment-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-positive {
    background: rgba(74, 222, 128, 0.12);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.25);
}
.badge-negative {
    background: rgba(248, 113, 113, 0.12);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.25);
}
.badge-neutral {
    background: rgba(148, 163, 184, 0.12);
    color: #94a3b8;
    border: 1px solid rgba(148, 163, 184, 0.25);
}
.badge-none {
    background: rgba(100, 100, 100, 0.12);
    color: #888;
    border: 1px solid rgba(100,100,100,0.2);
}

/* No aspect */
.no-aspect {
    font-family: 'DM Mono', monospace;
    color: #555;
    font-size: 0.85rem;
    text-align: center;
    padding: 1.5rem;
}

/* Stats row */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-box {
    flex: 1;
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.stat-number {
    font-size: 1.6rem;
    font-weight: 800;
    color: #c8a97e;
    line-height: 1;
}
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* Model selector */
.stSelectbox > div > div {
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #f0ece4 !important;
}

/* Info box */
.info-box {
    background: #161616;
    border-left: 3px solid #c8a97e;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #888;
    margin-bottom: 1.5rem;
}

/* Raw output */
.raw-output {
    background: #111;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #666;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Model loader ──────────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "Joint Task (Laptops + Restaurants)": "kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined",
    "ATE — Aspect Term Extraction": "kevinscaria/ate_tk-instruct-base-def-pos-neg-neut-combined",
    "ATSC — Sentiment Classification": "kevinscaria/atsc_tk-instruct-base-def-pos-combined",
}

INSTRUCTION = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life.
        output: battery life:positive, 
        Positive example 2-
        input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
        output: features:positive, iChat:positive, Photobooth:positive, garage band:positive
        Now complete the following example-
        input: """
EOS = " \noutput:"


@st.cache_resource(show_spinner=False)
def load_model(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device


def predict(text, tokenizer, model, device):
    prompt = INSTRUCTION + text + EOS
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def parse_output(raw):
    """Parse 'aspect:polarity, aspect:polarity' into list of dicts."""
    results = []
    if not raw or raw.strip() == "":
        return results
    pairs = [p.strip() for p in raw.split(",") if p.strip()]
    for pair in pairs:
        parts = pair.rsplit(":", 1)
        if len(parts) == 2:
            aspect, polarity = parts[0].strip(), parts[1].strip().lower()
        else:
            aspect, polarity = pair.strip(), "none"
        results.append({"aspect": aspect, "polarity": polarity})
    return results


def sentiment_badge(polarity):
    cls = {
        "positive": "badge-positive",
        "negative": "badge-negative",
        "neutral":  "badge-neutral",
    }.get(polarity, "badge-none")
    return f'<span class="sentiment-badge {cls}">{polarity}</span>'


# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">InstructABSA</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Aspect-Based Sentiment Analysis · NLP Final Project</div>', unsafe_allow_html=True)

st.markdown('<div class="info-box">Masukkan teks review produk atau restoran, model akan mengekstrak aspek beserta sentimennya secara otomatis.</div>', unsafe_allow_html=True)

# Model selector
model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
model_checkpoint = MODEL_OPTIONS[model_label]

# Load model
with st.spinner("Memuat model..."):
    tokenizer, model, device = load_model(model_checkpoint)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Input
user_input = st.text_area(
    "Teks Review",
    placeholder="Contoh: The battery life is amazing but the keyboard feels cheap and the screen is just okay.",
    height=120,
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze = st.button("Analisis Sentimen →")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Example buttons
st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#555;text-transform:uppercase;letter-spacing:0.1em;">Contoh cepat</p>', unsafe_allow_html=True)
examples = [
    "The battery life is amazing but the keyboard feels cheap.",
    "Great food and excellent service, but the ambience was a bit noisy.",
    "Screen resolution is top notch, however the price is way too high.",
]
ex_cols = st.columns(3)
for i, ex in enumerate(examples):
    with ex_cols[i]:
        if st.button(f"#{i+1}", key=f"ex_{i}", help=ex):
            user_input = ex
            analyze = True

# ── Inference ─────────────────────────────────────────────────────────────────
if analyze and user_input.strip():
    with st.spinner("Menganalisis..."):
        raw_output = predict(user_input.strip(), tokenizer, model, device)
        parsed = parse_output(raw_output)

    # Stats
    n_total = len(parsed)
    n_pos = sum(1 for p in parsed if p["polarity"] == "positive")
    n_neg = sum(1 for p in parsed if p["polarity"] == "negative")
    n_neu = sum(1 for p in parsed if p["polarity"] in ("neutral", "none"))

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-box"><div class="stat-number">{n_total}</div><div class="stat-label">Aspek</div></div>
        <div class="stat-box"><div class="stat-number" style="color:#4ade80">{n_pos}</div><div class="stat-label">Positif</div></div>
        <div class="stat-box"><div class="stat-number" style="color:#f87171">{n_neg}</div><div class="stat-label">Negatif</div></div>
        <div class="stat-box"><div class="stat-number" style="color:#94a3b8">{n_neu}</div><div class="stat-label">Netral</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Result cards
    if parsed:
        for item in parsed:
            asp = item["aspect"]
            pol = item["polarity"]
            if asp.lower() == "noaspectterm":
                st.markdown('<div class="no-aspect">Tidak ditemukan aspek dalam teks ini.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card">
                    <span class="aspect-term">{asp}</span>
                    {sentiment_badge(pol)}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="no-aspect">Tidak ada output dari model.</div>', unsafe_allow_html=True)

    with st.expander("Raw model output"):
        st.markdown(f'<div class="raw-output">{raw_output}</div>', unsafe_allow_html=True)

elif analyze and not user_input.strip():
    st.warning("Masukkan teks review terlebih dahulu.")

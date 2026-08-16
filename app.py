import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Ulasan SeaBank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CSS CUSTOM
# ==========================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }

    div[data-testid="stMetricLabel"] {
        color: #9CA3AF;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: white;
        font-size: 34px;
        font-weight: 700;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: white;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    hr {
        border: 0;
        height: 1px;
        background: #1F2937;
        margin: 24px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("seabank_reviews.csv")

    # Membersihkan nama kolom
    df.columns = [c.lower().strip() for c in df.columns]

    # Normalisasi label sentimen
    if "sentiment" in df.columns:
        df["sentiment"] = (
            df["sentiment"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "positive": "positif",
                "negative": "negatif",
                "neutral": "netral"
            })
        )

    # Konversi tanggal
    if "tanggal" in df.columns:
        df["tanggal"] = pd.to_datetime(
            df["tanggal"],
            errors="coerce"
        )

    # Konversi rating
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(
            df["rating"],
            errors="coerce"
        )

    return df


df = load_data()

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("⚙️ Filter Dashboard")

sentiment_options = sorted(
    df["sentiment"].dropna().unique()
)

selected_sentiment = st.sidebar.multiselect(
    "Pilih Sentimen",
    sentiment_options,
    default=sentiment_options
)

filtered = df[
    df["sentiment"].isin(selected_sentiment)
].copy()

# Filter rating
if (
    "rating" in filtered.columns
    and filtered["rating"].notna().any()
):

    rating_min = int(filtered["rating"].min())
    rating_max = int(filtered["rating"].max())

    selected_rating = st.sidebar.slider(
        "Filter Rating",
        rating_min,
        rating_max,
        (rating_min, rating_max)
    )

    filtered = filtered[
        (filtered["rating"] >= selected_rating[0])
        & (filtered["rating"] <= selected_rating[1])
    ]

# ==========================================================
# HEADER
# ==========================================================
st.title(
    "🏦 Dashboard Analisis Sentimen Ulasan SeaBank"
)

st.markdown(
    """
    Dashboard ini menyajikan visualisasi komprehensif
    hasil analisis sentimen terhadap **5.000 ulasan pengguna
    aplikasi SeaBank** sebagai representasi persepsi dan
    pengalaman pengguna berdasarkan data ulasan yang telah diproses.
    """
)

# ==========================================================
# KPI
# ==========================================================
total = len(filtered)

positive = (
    filtered["sentiment"] == "positif"
).sum()

negative = (
    filtered["sentiment"] == "negatif"
).sum()

neutral = (
    filtered["sentiment"] == "netral"
).sum()

if (
    "rating" in filtered.columns
    and filtered["rating"].notna().any()
):
    avg_rating = filtered["rating"].mean()
else:
    avg_rating = 0

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total Ulasan",
    f"{total:,}"
)

k2.metric(
    "Sentimen Positif",
    f"{positive:,}"
)

k3.metric(
    "Sentimen Negatif",
    f"{negative:,}"
)

k4.metric(
    "Rata-rata Rating",
    f"{avg_rating:.2f}"
)

st.markdown("---")

# ==========================================================
# DISTRIBUSI SENTIMEN & RATING
# ==========================================================
c1, c2 = st.columns(2)

# ----------------------------------------------------------
# DISTRIBUSI SENTIMEN
# ----------------------------------------------------------
with c1:

    st.markdown(
        '<div class="section-title">Distribusi Sentimen</div>',
        unsafe_allow_html=True
    )

    sentiment_count = (
        filtered["sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_count.columns = [
        "Sentimen",
        "Jumlah"
    ]

    fig_pie = px.pie(
        sentiment_count,
        names="Sentimen",
        values="Jumlah",
        hole=0.65,
        color="Sentimen",
        color_discrete_map={
            "positif": "#2563EB",
            "negatif": "#DC2626",
            "netral": "#F59E0B"
        }
    )

    fig_pie.update_layout(
        height=400,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        )
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

# ----------------------------------------------------------
# DISTRIBUSI RATING
# ----------------------------------------------------------
with c2:

    st.markdown(
        '<div class="section-title">Distribusi Rating</div>',
        unsafe_allow_html=True
    )

    rating_count = (
        filtered["rating"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    rating_count.columns = [
        "Rating",
        "Jumlah"
    ]

    fig_bar = px.bar(
        rating_count,
        x="Rating",
        y="Jumlah",
        color="Jumlah",
        color_continuous_scale="Blues"
    )

    fig_bar.update_layout(
        showlegend=False,
        height=400,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        )
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

st.markdown("---")

# ==========================================================
# TREN JUMLAH ULASAN HARIAN
# ==========================================================
st.markdown(
    '<div class="section-title">Tren Jumlah Ulasan Harian Berdasarkan Sentimen</div>',
    unsafe_allow_html=True
)

# Pastikan tanggal valid
trend_data = filtered.dropna(
    subset=["tanggal", "sentiment"]
).copy()

trend = (
    trend_data
    .groupby(
        [
            trend_data["tanggal"].dt.date,
            "sentiment"
        ]
    )
    .size()
    .reset_index(name="Jumlah")
)

trend.columns = [
    "Tanggal",
    "Sentimen",
    "Jumlah"
]

fig_line = px.line(
    trend,
    x="Tanggal",
    y="Jumlah",
    color="Sentimen",
    color_discrete_map={
        "positif": "#2563EB",
        "negatif": "#DC2626",
        "netral": "#F59E0B"
    }
)

fig_line.update_traces(
    mode="lines",
    line=dict(width=3)
)

fig_line.update_layout(
    xaxis_title="Tanggal",
    yaxis_title="Jumlah Ulasan",
    hovermode="x unified",
    legend_title="Sentimen",
    height=450,
    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10
    )
)

st.plotly_chart(
    fig_line,
    use_container_width=True
)

st.markdown("---")

# ==========================================================
# WORD CLOUD
# ==========================================================
st.markdown(
    '<div class="section-title">Word Cloud Ulasan</div>',
    unsafe_allow_html=True
)

wc1, wc2 = st.columns(2)


def make_wordcloud(text, cmap):

    if not text.strip():
        return None

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=cmap,
        max_words=150
    ).generate(text)

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.imshow(
        wc,
        interpolation="bilinear"
    )

    ax.axis("off")

    plt.tight_layout()

    return fig


# ----------------------------------------------------------
# WORD CLOUD POSITIF
# ----------------------------------------------------------
with wc1:

    st.subheader(
        "Word Cloud Positif"
    )

    pos_text = " ".join(
        filtered[
            filtered["sentiment"] == "positif"
        ]["review"]
        .dropna()
        .astype(str)
    )

    fig = make_wordcloud(
        pos_text,
        "Blues"
    )

    if fig:
        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    else:
        st.info(
            "Tidak terdapat ulasan positif."
        )


# ----------------------------------------------------------
# WORD CLOUD NEGATIF
# ----------------------------------------------------------
with wc2:

    st.subheader(
        "Word Cloud Negatif"
    )

    neg_text = " ".join(
        filtered[
            filtered["sentiment"] == "negatif"
        ]["review"]
        .dropna()
        .astype(str)
    )

    fig = make_wordcloud(
        neg_text,
        "Reds"
    )

    if fig:
        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    else:
        st.info(
            "Tidak terdapat ulasan negatif."
        )

st.markdown("---")

# ==========================================================
# TOP KATA
# ==========================================================
st.markdown(
    '<div class="section-title">10 Kata yang Paling Sering Muncul</div>',
    unsafe_allow_html=True
)


def top_words(texts, n=10):

    text = " ".join(
        texts
    ).lower()

    words = re.findall(
        r"[a-zA-ZÀ-ÿ]+",
        text
    )

    stopwords = {
        "yang",
        "dan",
        "di",
        "ke",
        "dari",
        "ini",
        "itu",
        "untuk",
        "pada",
        "dengan",
        "saya",
        "aku",
        "nya",
        "aja",
        "ga",
        "gak",
        "tidak",
        "ya",
        "kok",
        "lah",
        "seabank",
        "bank",
        "app",
        "aplikasi"
    }

    words = [
        w for w in words
        if len(w) > 2
        and w not in stopwords
    ]

    return Counter(
        words
    ).most_common(n)


t1, t2 = st.columns(2)

# ----------------------------------------------------------
# TOP KATA POSITIF
# ----------------------------------------------------------
with t1:

    st.subheader(
        "Top Kata Positif"
    )

    top_pos = pd.DataFrame(
        top_words(
            filtered[
                filtered["sentiment"] == "positif"
            ]["review"]
            .dropna()
            .astype(str)
        ),
        columns=[
            "Kata",
            "Frekuensi"
        ]
    )

    st.dataframe(
        top_pos,
        use_container_width=True,
        hide_index=True
    )


# ----------------------------------------------------------
# TOP KATA NEGATIF
# ----------------------------------------------------------
with t2:

    st.subheader(
        "Top Kata Negatif"
    )

    top_neg = pd.DataFrame(
        top_words(
            filtered[
                filtered["sentiment"] == "negatif"
            ]["review"]
            .dropna()
            .astype(str)
        ),
        columns=[
            "Kata",
            "Frekuensi"
        ]
    )

    st.dataframe(
        top_neg,
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# ==========================================================
# DATA ULASAN
# ==========================================================
st.markdown(
    '<div class="section-title">Data Ulasan</div>',
    unsafe_allow_html=True
)

display_columns = [
    "tanggal",
    "rating",
    "sentiment",
    "review"
]

available_columns = [
    col for col in display_columns
    if col in filtered.columns
]

st.dataframe(
    filtered[
        available_columns
    ].reset_index(drop=True),
    use_container_width=True,
    height=420,
    hide_index=True
)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")

st.markdown(
    """
    <div style='
        text-align: center;
        color: #9CA3AF;
        font-size: 14px;
        padding: 10px 0;
    '>
        Dashboard Analisis Sentimen Ulasan SeaBank
        • © 2026 • @Agil Firli Gunawan
    </div>
    """,
    unsafe_allow_html=True
)

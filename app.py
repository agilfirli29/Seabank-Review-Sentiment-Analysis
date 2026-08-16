# =========================================
# SEABANK SENTIMENT INTELLIGENCE
# STREAMLIT APPLICATION
# =========================================

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="SeaBank Sentiment Intelligence",
    page_icon="📊",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.main {
    background-color: #FFFFFF;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    color: #C62828;
}

.subtitle {
    font-size: 18px;
    color: #616161;
    margin-bottom: 20px;
}

.metric-card {
    background-color: #FAFAFA;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #E0E0E0;
}

.result-card {
    background-color: #F5F5F5;
    padding: 25px;
    border-radius: 15px;
    border-left: 8px solid #C62828;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL & TOKENIZER
# =========================================

model = load_model('model_lstm.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# =========================================
# STOPWORD & STEMMER
# =========================================

stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()

stemmer = factory.create_stemmer()

# =========================================
# TEXT PREPROCESSING FUNCTION
# =========================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'@\w+', '', text)

    text = re.sub(r'#\w+', '', text)

    text = re.sub(r'\d+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    text = ' '.join(tokens)

    return text

# =========================================
# LABEL MAPPING
# =========================================

label_reverse = {
    0: "Negative 😡",
    1: "Neutral 😐",
    2: "Positive 😊"
}

# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("📌 Project Information")

st.sidebar.markdown("""
### About

This application analyzes user reviews
from SeaBank Google Play Store reviews
using Deep Learning LSTM.
""")

st.sidebar.markdown("### 🤖 Model")
st.sidebar.write("LSTM Deep Learning")

st.sidebar.markdown("### 📊 Classification")
st.sidebar.write("""
- Positive
- Neutral
- Negative
""")

st.sidebar.markdown("### 🗂 Dataset")
st.sidebar.write("10,000+ Google Play Store Reviews")

st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Agil Firli Gunawan")

# =========================================
# HEADER
# =========================================

st.markdown(
    """
    <div class="title">
        SeaBank Sentiment Intelligence
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI-Powered Sentiment Analysis for Digital Banking Reviews
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================
# METRICS
# =========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>Dataset</h3>
        <h2>10K+</h2>
        <p>Reviews</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>Model</h3>
        <h2>LSTM</h2>
        <p>Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>Classes</h3>
        <h2>3</h2>
        <p>Sentiments</p>
    </div>
    """, unsafe_allow_html=True)

# =========================================
# MAIN CONTENT
# =========================================

st.markdown("---")

left_col, right_col = st.columns([2,1])

# =========================================
# INPUT SECTION
# =========================================

with left_col:

    st.subheader("✍️ Input Review")

    user_input = st.text_area(
        "",
        height=220,
        placeholder="Example: Aplikasi sangat membantu dan transfer cepat"
    )

    analyze_button = st.button(
        "🔍 Analyze Sentiment"
    )

# =========================================
# RESULT SECTION
# =========================================

with right_col:

    st.subheader("📊 Prediction Result")

    if analyze_button:

        if user_input.strip() == "":

            st.warning(
                "Please enter a review first."
            )

        else:

            # Preprocessing
            cleaned_text = clean_text(user_input)

            # Convert to sequence
            sequence = tokenizer.texts_to_sequences(
                [cleaned_text]
            )

            # Padding
            padded = pad_sequences(
                sequence,
                maxlen=100
            )

            # Prediction
            prediction = model.predict(padded)

            predicted_class = np.argmax(prediction)

            confidence = np.max(prediction) * 100

            sentiment = label_reverse[predicted_class]

            st.markdown(
                f"""
                <div class="result-card">
                    <h2>{sentiment}</h2>
                    <hr>
                    <h4>Confidence Score</h4>
                    <h2>{confidence:.2f}%</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================================
# ABOUT SECTION
# =========================================

st.markdown("---")

st.subheader("📖 About This Project")

st.write("""
This project was developed to analyze user sentiment
towards the SeaBank mobile banking application based
on Google Play Store reviews.

The sentiment classification uses Deep Learning LSTM
to classify reviews into:
- Positive
- Neutral
- Negative

This project aims to provide insights into user
satisfaction and digital banking service quality.
""")

# =========================================
# FOOTER
# =========================================

st.markdown(
    """
    <div class="footer">
        Developed by Agil Firli Gunawan
    </div>
    """,
    unsafe_allow_html=True
)

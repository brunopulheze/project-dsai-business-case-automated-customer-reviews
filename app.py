"""
Streamlit App — Automated Customer Review System

Run with: streamlit run app.py

This app combines the three project components:
1. Review Classification — classify a review as Positive, Neutral, or Negative
2. Product Clustering — explore product meta-categories
3. Review Summarization — read generated recommendation articles
"""

import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ─────────────────────────────────────────────
# App configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Automated Customer Reviews",
    page_icon="⭐",
    layout="wide",
)

st.title("⭐ Automated Customer Review System")
st.markdown("Classify reviews, explore product clusters, and read AI-generated recommendation articles.")


# ─────────────────────────────────────────────
# Helper: load models (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    """Load the fine-tuned sentiment classification model."""
    model_path = "models/sentiment_classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model


@st.cache_resource
def load_summarizer():
    """Load the summarization pipeline."""
    return pipeline("summarization", model="facebook/bart-large-cnn")


@st.cache_data
def load_data():
    """Load the clustered reviews dataframe."""
    return pd.read_csv("data/clustered_reviews.csv")


@st.cache_data
def load_articles():
    """Load generated recommendation articles."""
    try:
        with open("data/recommendation_articles.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📝 Review Classification", "📦 Product Clusters", "📰 Recommendation Articles"],
)


# ─────────────────────────────────────────────
# Page: Home
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.header("Welcome!")
    st.markdown(
        """
        This app showcases an NLP-powered product review analysis system built
        for **Amazon product reviews**. Use the sidebar to explore:
        
        - **Review Classification** — Enter a review and get a sentiment prediction.
        - **Product Clusters** — Browse products organized into meta-categories.
        - **Recommendation Articles** — Read AI-generated blog posts with product recommendations.
        """
    )


# ─────────────────────────────────────────────
# Page: Review Classification
# ─────────────────────────────────────────────
elif page == "📝 Review Classification":
    st.header("📝 Review Classification")
    st.markdown("Enter a product review below to classify its sentiment.")

    review_text = st.text_area(
        "Paste a review here:",
        height=150,
        placeholder="e.g. 'This tablet is amazing! Great battery life and the screen is crystal clear.'",
    )

    if st.button("Classify", type="primary"):
        if review_text.strip():
            with st.spinner("Classifying..."):
                try:
                    tokenizer, model = load_classifier()
                    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, max_length=256)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                    labels = ["Negative", "Neutral", "Positive"]
                    pred_idx = torch.argmax(probs).item()

                    st.success(f"**Predicted Sentiment: {labels[pred_idx]}**")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Negative", f"{probs[0]:.1%}")
                    col2.metric("Neutral", f"{probs[1]:.1%}")
                    col3.metric("Positive", f"{probs[2]:.1%}")
                except Exception as e:
                    st.error(f"Error loading model. Make sure the model is saved in `models/sentiment_classifier/`. Details: {e}")
        else:
            st.warning("Please enter a review.")


# ─────────────────────────────────────────────
# Page: Product Clusters
# ─────────────────────────────────────────────
elif page == "📦 Product Clusters":
    st.header("📦 Product Clusters")
    st.markdown("Explore products grouped into meta-categories.")

    try:
        df = load_data()

        categories = sorted(df["meta_category"].dropna().unique())
        selected_cat = st.selectbox("Select a category:", categories)

        cat_df = df[df["meta_category"] == selected_cat]
        st.write(f"**{len(cat_df)} reviews** in this category")

        # Product-level summary
        product_summary = (
            cat_df.groupby("name")
            .agg(
                avg_rating=("reviews.rating", "mean"),
                num_reviews=("reviews.rating", "count"),
            )
            .reset_index()
            .sort_values("avg_rating", ascending=False)
        )
        st.dataframe(product_summary, use_container_width=True)

        # Sentiment breakdown
        if "sentiment" in cat_df.columns:
            st.subheader("Sentiment Breakdown")
            sentiment_counts = cat_df["sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

    except FileNotFoundError:
        st.warning("Clustered data not found. Run the clustering notebook first to generate `data/clustered_reviews.csv`.")
    except Exception as e:
        st.error(f"Error: {e}")


# ─────────────────────────────────────────────
# Page: Recommendation Articles
# ─────────────────────────────────────────────
elif page == "📰 Recommendation Articles":
    st.header("📰 AI-Generated Recommendation Articles")

    articles_text = load_articles()

    if articles_text:
        st.markdown(articles_text)
    else:
        st.info(
            "No articles found. Run the summarization notebook first to generate "
            "`data/recommendation_articles.txt`."
        )

        st.markdown("---")
        st.subheader("Or generate a summary on the fly")
        category_input = st.text_input("Enter a category name:")
        reviews_input = st.text_area("Paste some reviews (one per line):", height=150)

        if st.button("Generate Summary", type="primary"):
            if category_input and reviews_input:
                with st.spinner("Generating..."):
                    try:
                        summarizer = load_summarizer()
                        prompt = f"Summarize the following {category_input} product reviews:\n{reviews_input}"
                        result = summarizer(prompt, max_length=300, min_length=80, do_sample=False)
                        st.write(result[0]["summary_text"])
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter both a category and reviews.")

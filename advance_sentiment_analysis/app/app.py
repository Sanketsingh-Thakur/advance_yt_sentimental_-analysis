# app/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_models, clean_text, fetch_comments

# Load models once
sentiment_model, ctype_model, vectorizer = load_models()

# Streamlit page config
st.set_page_config(page_title="YouTube Comment Analyzer", layout="wide")

# --- Header ---
st.title("🎥 YouTube Comment Analyzer")
st.markdown(
    """
    Analyze YouTube video comments automatically!  
    Paste a video link, fetch comments, and classify them as **Positive/Negative/Neutral**  
    and **Request/Suggestion/Other** using your trained ML model. 🚀
    """
)

# --- Input ---
video_url = st.text_input("🔗 Enter a YouTube video URL:")
num_comments = st.slider("How many comments to fetch?", min_value=10, max_value=200, value=50, step=10)

if st.button("🔍 Analyze Comments"):
    if video_url.strip() == "":
        st.warning("⚠️ Please enter a YouTube video URL.")
    else:
        with st.spinner("Fetching comments..."):
            comments = fetch_comments(video_url, limit=num_comments)

        if not comments:
            st.error("❌ Could not fetch comments. Maybe comments are disabled?")
        else:
            st.success(f"✅ Fetched {len(comments)} comments!")

            # Process comments
            cleaned = [clean_text(c) for c in comments]
            X_vec = vectorizer.transform(cleaned)

            sentiments = sentiment_model.predict(X_vec)
            ctypes = ctype_model.predict(X_vec)

            # Create dataframe
            df = pd.DataFrame({
                "Comment": comments,
                "Sentiment": sentiments,
                "Comment Type": ctypes
            })

            # --- Results Table ---
            st.subheader("📋 Analyzed Comments")
            st.dataframe(df, use_container_width=True)

            # --- Download Button ---
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name="youtube_comments_analysis.csv",
                mime="text/csv"
            )

            # --- Summary ---
            st.subheader("📊 Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Sentiment Distribution (Bar)")
                st.bar_chart(df["Sentiment"].value_counts())

            with col2:
                st.markdown("### Comment Type Distribution (Bar)")
                st.bar_chart(df["Comment Type"].value_counts())

            # --- Pie Charts ---
            st.subheader("🥧 Overall Analysis")

            col3, col4 = st.columns(2)

            with col3:
                sentiment_counts = df["Sentiment"].value_counts()
                fig1, ax1 = plt.subplots()
                ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90)
                ax1.axis("equal")
                st.pyplot(fig1)

            with col4:
                ctype_counts = df["Comment Type"].value_counts()
                fig2, ax2 = plt.subplots()
                ax2.pie(ctype_counts, labels=ctype_counts.index, autopct="%1.1f%%", startangle=90)
                ax2.axis("equal")
                st.pyplot(fig2)

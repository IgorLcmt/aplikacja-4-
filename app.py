import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import requests
import pandas as pd
from utils import (
    is_valid_url,
    load_database,
    filter_dataframe,
    get_industry_similarity,
    get_industry_embeddings,
    embed_text_batch,
    find_most_similar_industry,
    paraphrase_query,
    cosine_similarity,
    init_openai
)

MAX_TEXT_LENGTH = 4000

st.set_page_config(page_title="AI Company Matcher", layout="wide")
st.title("🤖 AI Company Matcher")

# --- Cached scraper ---
@st.cache_data(show_spinner="Fetching website...")
def scrape_website_cached(url: str) -> str:
    if not is_valid_url(url):
        return ""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text() for p in paragraphs)
        return text[:MAX_TEXT_LENGTH]
    except Exception:
        return ""

# --- Summary logic ---
def get_summarized_website(url: str, client: OpenAI) -> str:
    scraped = scrape_website_cached(url)
    if scraped:
        return summarize_scraped_text(scraped, client)
    return ""

# --- Main logic ---
def main():
    # OpenAI key check
    api_key = st.secrets.get("openai", {}).get("api_key")
    if not api_key:
        st.error("Missing OpenAI API key.")
        st.stop()
    client = OpenAI(api_key=api_key)

    st.sidebar.header("🔍 Input Options")
    query_input = st.text_input("Enter company website URL:")
    text_input = st.text_area("Optional extra description:")
    database = load_database()

    # Industry selector
    industries = sorted(database["Primary Industry"].dropna().unique())
    selected_industry = st.sidebar.selectbox("Filter by industry (optional)", [""] + industries)
    use_detected_also = st.sidebar.checkbox("Include detected industry in filtering", value=False)

    query_text = ""
    summarized = ""

    if query_input and is_valid_url(query_input):
        with st.spinner("Scraping and summarizing website..."):
            summarized = get_summarized_website(query_input, client)
            if summarized.strip():
                summarized = st.text_area(
                    "📝 Website Summary (you can edit it before matching):",
                    summarized,
                    height=250
                )
                query_text += summarized
            else:
                st.warning("Could not extract usable content from the website.")

    if text_input.strip():
        query_text += "\n" + text_input.strip()

    st.markdown("### 🔎 Match Companies")

    if st.button("🔍 Find Matches"):
        if not query_text.strip():
            st.error("Please enter a valid input (description or website).")
            return

        st.info("Analyzing description and preparing matches...")

        # Embed and paraphrase
        query_variants = [query_text] + paraphrase_query(query_text, client)
        query_embeddings = embed_text_batch(query_variants, client)
        query_avg_embedding = sum(query_embeddings) / len(query_embeddings)

        # Filter database
        df_filtered = filter_dataframe(database, selected_industry, use_detected_also, query_text, client)
        if df_filtered.empty:
            st.warning("No companies matched your filters.")
            return

        # Similarity matching
        company_embeddings = list(df_filtered["embedding"])
        similarities = cosine_similarity(query_avg_embedding, company_embeddings)
        df_filtered["score"] = similarities
        df_top = df_filtered.sort_values("score", ascending=False).head(10)

        st.success(f"Top {len(df_top)} company matches:")
        st.dataframe(df_top[["Company", "Primary Industry", "score"]])

if __name__ == "__main__":
    main()

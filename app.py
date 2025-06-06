import streamlit as st
from utils import (
    load_database,
    embed_text_batch,
    cosine_similarity,
    summarize_scraped_text,
    detect_industry_from_text
)
from openai import OpenAI

def main():
    st.set_page_config(page_title="CMT Company Analyzer", layout="wide")
    st.title("CMT Company Analyzer")

    api_key = st.secrets.get("openai", {}).get("api_key", "").strip()
    if not api_key:
        st.error("Missing OpenAI API key.")
        st.stop()

    client = OpenAI(api_key=api_key)

    st.sidebar.subheader("Input Company Details")
    website = st.sidebar.text_input("🌐 Company website (optional):", "")
    manual_desc = st.sidebar.text_area("📝 Company description (optional):")
    min_value = float(st.sidebar.text_input("Min Enterprise Value ($M)", "0").replace(",", "."))
    max_value = float(st.sidebar.text_input("Max Enterprise Value ($M)", "10000").replace(",", "."))
    include_ai_filter = st.sidebar.checkbox("Include detected industry in filtering", value=False)
    df, industry_list = load_database()
    industry_selection = st.sidebar.multiselect("🎯 Filter by industry (optional):", industry_list)

    if website or manual_desc:
        with st.spinner("Analyzing company profile..."):
            scraped = summarize_scraped_text(website, client) if website else ""
            base_desc = manual_desc.strip() or scraped.strip()
            st.subheader("📋 Website Summary (you can edit it before matching):")
            query_input = st.text_area("✏️ Website Summary:", base_desc, height=200)
            detected_industry = detect_industry_from_text(query_input, client)
            st.info(f"Detected primary industry: **{detected_industry}**")
            st.info("✅ Description processed. Matching logic can go here.")

if __name__ == "__main__":
    main()

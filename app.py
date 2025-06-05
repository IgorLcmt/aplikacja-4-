import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import re
import time
import io
import html
import validators
import bleach

# ===== Constants =====
MAX_TEXT_LENGTH = 4000
MAX_TOKENS = 8000
BATCH_SIZE = 100

# ===== Streamlit Config =====
st.set_page_config(page_title="CMT Transaction Finder", layout="wide")
st.title("CMT Transaction Finder 🔍")
st.caption("Find comparable M&A transactions for valuation analysis.")

# ===== Utilities =====
def clean_text_input(text: str) -> str:
    safe_text = html.escape(text.strip())[:2000]
    return bleach.clean(safe_text, tags=[], strip=True)

def is_valid_url(url: str) -> bool:
    return isinstance(url, str) and validators.url(url.strip())

def truncate_text(text: str, encoding_name: str = "cl100k_base") -> str:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)[:MAX_TOKENS]
    return encoding.decode(tokens)

@st.cache_data
def scrape_website(url: str) -> str:
    if not is_valid_url(url):
        return ""
    try:
        time.sleep(1.0)
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:MAX_TEXT_LENGTH]
    except Exception:
        return ""

@st.cache_data
def load_database() -> pd.DataFrame:
    try:
        df = pd.read_excel("app_data/Database.xlsx", engine="openpyxl")
        df.columns = [col.strip().replace('\xa0', ' ') for col in df.columns]
        val_col = "Total Enterprise Value (mln$)"
        if val_col in df.columns:
            df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Database loading failed: {str(e)}")
        st.stop()

@st.cache_resource
def init_openai(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def gpt_chat(system_prompt: str, user_prompt: str, client: OpenAI) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return ""

def detect_industry_from_text(text: str, client: OpenAI) -> str:
    return gpt_chat(
        "Identify the company's primary industry (e.g. IT, Healthcare, Retail). Respond with one word.",
        text,
        client
    )

def generate_tags(description: str, client: OpenAI) -> str:
    return gpt_chat("Extract 3-5 high-level keywords or tags that summarize this company.", description, client)

def paraphrase_query(query: str, client: OpenAI) -> List[str]:
    output = gpt_chat("Paraphrase this business description into 3 alternate versions.", query, client)
    return [line.strip("-• ") for line in output.splitlines() if line.strip()]

def explain_match(query: str, company_desc: str, client: OpenAI) -> str:
    prompt = f"""
Based on the provided business description and the target profile, explain in 3–5 bullet points why this transaction is a good match.

Focus on:
1. Industry
2. Product or service type
3. Sales channels (B2B/B2C)
4. Customer segment

Query:
{query}

Company Description:
{company_desc}
"""
    return gpt_chat("You are a factual M&A assistant.", prompt, client)

@st.cache_data
def embed_text_batch(texts: List[str], client: OpenAI) -> List[List[float]]:
    from numpy import array
    clean_texts = [truncate_text(t.strip()) for t in texts if isinstance(t, str) and t.strip()]
    if not clean_texts:
        return []
    embeddings = []
    for i in range(0, len(clean_texts), BATCH_SIZE):
        batch = clean_texts[i:i + BATCH_SIZE]
        response = client.embeddings.create(input=batch, model="text-embedding-3-small")
        embeddings.extend([record.embedding for record in response.data])
    return embeddings

@st.cache_data(show_spinner="Embedding company descriptions...")
def get_cached_db_embeddings(descriptions: List[str], client: OpenAI) -> np.ndarray:
    return np.array(embed_text_batch(descriptions, client))

# ===== Main App Logic =====
def main():
    api_key = st.secrets.get("openai", {}).get("api_key")
    if not api_key:
        st.error("Missing OpenAI API key.")
        st.stop()
    client = init_openai(api_key)

    with st.sidebar:
        raw_url = st.text_input("🌐 Company website (optional):")
        raw_description = st.text_area("📝 Business description (optional):")
        min_value = st.number_input("Min Enterprise Value ($M)", 0.0, value=0.0, step=10.0)
        max_value = st.number_input("Max Enterprise Value ($M)", 0.0, value=10000.0, step=10.0)
        start_search = st.button("🔍 Find Matches")

    if not start_search:
        st.info("Enter a company website or description to begin.")
        return

    # Combine Inputs
    query_text = ""
    if raw_url:
        scraped = scrape_website(raw_url)
        if scraped:
            query_text += scraped
    if raw_description:
        query_text = raw_description.strip() + "\n" + query_text

    if not query_text.strip():
        st.error("Please provide a valid input.")
        return

    # Keyword Enrichment
    with st.spinner("Extracting tags..."):
        tags = generate_tags(query_text, client)
        query_text += f"\nTags: {tags}"

    with st.spinner("Detecting industry..."):
        detected_industry = detect_industry_from_text(query_text, client)

    df = load_database()

    if "Primary Industry" in df.columns:
        df_filtered = df[df["Primary Industry"].str.contains(detected_industry, case=False, na=False)]
        if df_filtered.empty:
            st.warning(f"No companies found in industry '{detected_industry}'. Showing all instead.")
        else:
            df = df_filtered

    if "Total Enterprise Value (mln$)" in df.columns:
        df = df[(df["Total Enterprise Value (mln$)"].fillna(0.0) >= min_value) &
                (df["Total Enterprise Value (mln$)"].fillna(float("inf")) <= max_value)]

    if df.empty:
        st.error("No companies match your filters.")
        return

         # Embedding and Matching
    descriptions = df["Business Description"].astype(str).tolist()

    # Defensive handling of paraphrasing
    paraphrases = paraphrase_query(query_text, client)
    if not paraphrases:
        paraphrases = []

    query_variants = [q.strip() for q in ([query_text] + paraphrases) if q.strip()]
    if not query_variants:
        st.error("Failed to generate valid query variants.")
        return

    query_embeds = np.array(embed_text_batch(query_variants, client))
    if query_embeds.size == 0:
        st.error("Failed to embed query.")
        return

    query_embed = np.average(query_embeds, axis=0).reshape(1, -1)
    db_embeds = get_cached_db_embeddings(descriptions, client)
    scores = cosine_similarity(db_embeds, query_embed).flatten()
    top_indices = np.argsort(-scores)[:20]
    df_top = df.iloc[top_indices].copy()

    with ThreadPoolExecutor() as executor:
        explanations = list(executor.map(lambda desc: explain_match(query_text, desc, client),
                                         df_top["Business Description"]))

    df_top["Similarity Score"] = scores[top_indices]
    df_top["Explanation"] = explanations
    df_top = df_top.sort_values("Similarity Score", ascending=False)

    # Display Results
    st.subheader("Top 20 Matching Transactions")
    st.dataframe(df_top[[
        "Target/Issuer Name", "Primary Industry", "Total Enterprise Value (mln$)",
        "Business Description", "Similarity Score", "Explanation"
    ]], use_container_width=True)
    
    # Download
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_top.to_excel(writer, index=False)
    st.download_button("⬇️ Download Results", data=output.getvalue(), file_name="Matches.xlsx")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_exponential, stop_after_attempt
import tiktoken
import io
import re
import time
import validators
import html
import bleach

# ===== Constants =====
MAX_TEXT_LENGTH = 4000
BATCH_SIZE = 100
MAX_TOKENS = 8000
SIMILARITY_THRESHOLD = 0.5

# ===== Streamlit Config =====
st.set_page_config(page_title="CMT Company Analyzer 🔍", layout="wide")
st.title("CMT Company Analyzer 🔍")
st.caption("An intelligent M&A transaction matcher")

# ===== Utility Functions =====
def clean_url(url: str) -> str:
    return url.strip() if validators.url(url.strip()) else ""

def clean_text_input(text: str) -> str:
    safe_text = html.escape(text.strip())[:2000]
    return bleach.clean(safe_text, tags=[], strip=True)

@st.cache_resource
def init_openai(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

@st.cache_data(show_spinner="Loading database...")
def load_database() -> pd.DataFrame:
    try:
        df = pd.read_excel("app_data/Database.xlsx", engine="openpyxl")
        df.columns = [col.strip().replace('\xa0', ' ') for col in df.columns]
        val_col = "Total Enterprise Value (mln$)"
        if val_col in df.columns:
            df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        required_cols = [
            'Target/Issuer Name', 'MI Transaction ID',
            'Implied Enterprise Value/ EBITDA (x)', 'Total Enterprise Value (mln$)',
            'Announcement Date', 'Company Geography (Target/Issuer)',
            'Business Description', 'Primary Industry', 'Web page'
        ]
        actual_cols = [col.strip().lower() for col in df.columns]
        required_check = [col.strip().lower() for col in required_cols]
        missing_required = [col for col in required_check if col not in actual_cols]
        if missing_required:
            st.error(f"Missing required columns: {missing_required}")
            st.stop()
        return df
    except Exception as e:
        st.error(f"Database loading failed: {str(e)}")
        st.stop()

def truncate_text(text: str, encoding_name="cl100k_base") -> str:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)[:MAX_TOKENS]
    return encoding.decode(tokens)

@st.cache_data
def embed_text_batch(texts: List[str], _client: OpenAI) -> List[List[float]]:
    clean_texts = [truncate_text(t.strip()) for t in texts if isinstance(t, str) and t.strip()]
    embeddings = []
    try:
        for i in range(0, len(clean_texts), BATCH_SIZE):
            batch = clean_texts[i:i + BATCH_SIZE]
            response = _client.embeddings.create(input=batch, model="text-embedding-3-small")
            embeddings.extend([record.embedding for record in response.data])
    except Exception as e:
        st.error(f"Embedding failed: {str(e)}")
        st.stop()
    return embeddings

@st.cache_data(show_spinner="Embedding database (cached)...")
def get_cached_db_embeddings(descriptions: List[str], _client: OpenAI) -> np.ndarray:
    return np.array(embed_text_batch(descriptions, _client))

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
        return response.choices[0].message.content.strip() if response and response.choices else ""
    except Exception as e:
        st.warning(f"GPT call failed: {str(e)}")
        return ""

def paraphrase_query(query: str, client: OpenAI) -> List[str]:
    response = gpt_chat("Paraphrase this business query into 3 alternate versions.", query, client)
    return [line.strip("-• ") for line in response.splitlines() if line.strip()]

def explain_match(query: str, company_desc: str, client: OpenAI) -> str:
    prompt = f"""
Based on the provided business description and the target profile, explain in 3–5 bullet points why this transaction is a good match.

Focus on:
- Industry
- Product type
- Sales channels
- Customer segments

Query:
{query}

Company:
{company_desc}
"""
    return gpt_chat("You are a factual M&A assistant.", prompt, client)

def detect_industry_from_text(text: str, client: OpenAI) -> str:
    return gpt_chat(
        "Identify the primary industry (Advertising, Healthcare, IT, etc.):", text, client
    )

@st.cache_data
def scrape_and_cache(url: str) -> str:
    try:
        if not re.match(r'^https?://', url):
            return ""
        time.sleep(1.0)
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:MAX_TEXT_LENGTH]
    except Exception:
        return ""

def get_top_indices(scores: np.ndarray, threshold: float) -> np.ndarray:
    qualified = scores >= threshold
    return np.argsort(-scores[qualified])

# ===== MAIN APP =====
def main():
    session_defaults = {
        "results": None,
        "scraped_cache": {},
        "previous_matches": set(),
        "generate_new": True
    }
    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ✅ API KEY CHECK
    api_key = st.secrets.get("openai", {}).get("api_key")
    if not api_key:
        st.error("OpenAI API key missing. Please add it to secrets.")
        st.stop()
    client = OpenAI(api_key=api_key)

    with st.sidebar:
        st.header("Search Criteria")
        raw_url = st.text_input("🌐 Company Website URL (optional):")
        query_input = clean_url(raw_url)
        raw_description = st.text_area("📝 Or enter description manually:")
        manual_description = clean_text_input(raw_description)
        st.markdown("### 💰 Transaction Size")
        min_value = st.number_input("Minimum Value ($M)", 0.0, 1e6, 0.0, 10.0)
        max_value = st.number_input("Maximum Value ($M)", 0.0, 1e6, 10000.0, 10.0)
        start_search = st.button("🔍 Find Matches")
        if st.button("🔄 Restart"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

    if not start_search and st.session_state.get("generate_new", True):
        st.info("Enter a website and/or description and click **Find Matches**.")
        return

    query_text = ""
    if start_search or st.session_state.get("generate_new", False):
        if query_input:
            with st.spinner("Scraping website..."):
                query_text = scrape_and_cache(query_input)
                if not query_text:
                    st.warning("Could not extract content from URL.")
    if manual_description:
        query_text = manual_description + "\n" + query_text
    if not query_text.strip():
        st.info("Please provide a valid company description or website.")
        return

    if st.session_state.generate_new:
        with st.spinner("Analyzing..."):
            try:
                df = load_database()
                detected_industry = detect_industry_from_text(query_text, client)
                if detected_industry:
                    df = df[df["Primary Industry"].str.contains(detected_industry, case=False, na=False)]
                st.sidebar.markdown(f"**🧠 Detected Industry:** {detected_industry or 'Unknown'}")

                df = df[
                    (df["Total Enterprise Value (mln$)"].fillna(0) >= min_value) &
                    (df["Total Enterprise Value (mln$)"].fillna(float("inf")) <= max_value)
                ]
                if df.empty:
                    st.warning("No companies match filters. Showing all.")
                    df = load_database()

                descriptions = df["Business Description"].astype(str).tolist()
                query_variants.append(detect_industry_from_text(query_text, client))
                db_embeds = get_cached_db_embeddings(descriptions, client)
                query_variants = [q.strip() for q in ([query_text] + paraphrase_query(query_text, client)) if q.strip()]
                if not query_variants:
                    st.error("No valid query variants generated.")
                    st.stop()

                query_embeds = np.array(embed_text_batch(query_variants, client))
                if query_embeds.size == 0:
                    st.error("Query embedding failed or returned empty result.")
                    st.stop()
                
                query_embed = np.average(query_embeds, axis=0, weights=[2.0] + [1.0]*(len(query_embeds)-1)).reshape(1, -1)
                scores = cosine_similarity(db_embeds, query_embed).flatten()
                top_indices = get_top_indices(scores, SIMILARITY_THRESHOLD)[:40]
                df_top40 = df.iloc[top_indices].copy()

                with ThreadPoolExecutor() as executor:
                    scraped_texts = list(executor.map(scrape_and_cache, df_top40["Web page"]))
                    explanations = list(executor.map(lambda desc: explain_match(query_text, desc, client),
                                                    df_top40["Business Description"]))

                df_top40["Explanation"] = explanations
                full_texts = [f"{desc}\n{text}" for desc, text in zip(df_top40["Business Description"], scraped_texts)]
                final_embeds = np.array(embed_text_batch(full_texts + [query_text], client)[:-1])
                final_scores = cosine_similarity(final_embeds, query_embed).flatten()
                df_top40["Score"] = final_scores
                df_top40["ID"] = df_top40["MI Transaction ID"].astype(str)
                df_final = df_top40[~df_top40["ID"].isin(st.session_state.previous_matches)].nlargest(20, "Score")
                st.session_state.previous_matches.update(df_final["ID"].tolist())
                st.session_state.results = df_final
                st.session_state.generate_new = False
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                st.stop()

    if st.session_state.results is not None:
        st.subheader("Top Matches")
        st.dataframe(st.session_state.results, use_container_width=True)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            st.session_state.results.to_excel(writer, index=False)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download Excel", data=output.getvalue(), file_name="Company_Matches.xlsx")
        with col2:
            if st.button("🔄 Find New Matches"):
                st.session_state.generate_new = True
                st.rerun()

if __name__ == "__main__":
    main()

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
import io
import re
import time

# ===== CONSTANTS =====
MAX_TEXT_LENGTH = 4000
BATCH_SIZE = 100
MAX_TOKENS = 8000
RATE_LIMIT_DELAY = 1.0
SIMILARITY_THRESHOLD = 0.5

# ===== STREAMLIT CONFIG =====
st.set_page_config(page_title="CMT Company Analyzer 🔍", layout="wide")

# ===== INITIALIZATION =====
@st.cache_resource
def init_openai(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

# ===== DATA LOADING =====
@st.cache_data(show_spinner="Loading database...")
def load_database() -> tuple[pd.DataFrame, list]:
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

        industry_list = sorted(df["Primary Industry"].dropna().unique())

        # ✅ Extract only the part inside parentheses
        raw_industries = df["Primary Industry"].dropna().unique().tolist()
        cleaned_industries = []
    
        for entry in raw_industries:
            match = re.search(r"\((.*?)\)", str(entry))
            if match:
                cleaned_industries.append(match.group(1).strip())
    
        # Remove duplicates and sort
        industry_list = sorted(set(cleaned_industries))
    
        return df, industry_list

    except Exception as e:
        st.error(f"Database loading failed: {str(e)}")
        st.stop()

# ===== EMBEDDING =====
def truncate_text(text: str, encoding_name: str = "cl100k_base") -> str:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)[:MAX_TOKENS]
    return encoding.decode(tokens)

@st.cache_data
def embed_text_batch(texts: List[str], _client: OpenAI) -> List[List[float]]:
    clean_texts = [truncate_text(t.strip()) for t in texts if isinstance(t, str)]
    embeddings = []
    try:
        for i in range(0, len(clean_texts), BATCH_SIZE):
            batch = clean_texts[i:i + BATCH_SIZE]
            response = _client.embeddings.create(input=batch, model="text-embedding-ada-002")
            embeddings.extend([record.embedding for record in response.data])
    except Exception as e:
        st.error(f"Embedding failed: {str(e)}")
        st.stop()
    return embeddings

# ===== GPT FUNCTIONS =====
def gpt_chat(system_prompt: str, user_prompt: str, client: OpenAI) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip() if response and response.choices else ""
    except Exception as e:
        st.warning(f"GPT call failed: {str(e)}")
        return ""

def explain_match(query: str, company_desc: str, client: OpenAI) -> str:
    return gpt_chat(
        "You are a business analyst evaluating company fit for strategic transactions.",
        f"""
Based on the provided business description and the target profile, explain in 3–5 bullet points why this transaction is a good match.

Focus on:
1. Industry
2. Product or service type
3. Sales channels
4. Customer segments

Query Profile:
{query}

Company Description:
{company_desc}
        """,
        client
    )

def generate_tags(description: str, client: OpenAI) -> str:
    return gpt_chat("Extract 3–5 high-level tags or categories from the business description.", description, client)

def paraphrase_query(query: str, client: OpenAI) -> List[str]:
    response = gpt_chat("Paraphrase this business query into 3 alternate versions.", query, client)
    return [line.strip("-• ") for line in response.splitlines() if line.strip()]

def detect_industry_from_text(text: str, client: OpenAI) -> str:
    return gpt_chat(
        "Based on the following text, identify the company's primary industry. Respond with one broad label like Advertising, Healthcare, Manufacturing, IT, etc.",
        text, client
    )

# ===== SCRAPING =====
def is_valid_url(url: str) -> bool:
    return isinstance(url, str) and re.match(r'^https?://', url) is not None

def scrape_website(url: str) -> str:
    if not is_valid_url(url):
        return ""
    try:
        time.sleep(RATE_LIMIT_DELAY)
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:MAX_TEXT_LENGTH]
    except Exception:
        return ""

# ===== UTILS =====
def get_top_indices(scores: np.ndarray, threshold: float) -> np.ndarray:
    qualified = scores >= threshold
    return np.argsort(-scores[qualified])

# ===== MAIN APP =====
def main():
    api_key = st.secrets.get("openai", {}).get("api_key")
    if not api_key:
        st.error("OpenAI API key missing")
        st.stop()
    client = OpenAI(api_key=api_key)

    session_defaults = {
        "results": None,
        "scraped_cache": {},
        "previous_matches": set(),
        "generate_new": True
    }
    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    df, industry_list = load_database()
    
    with st.sidebar:
        query_input = st.text_input("🌐 Paste company website URL (optional):")
        manual_description = st.text_area("📝 Or provide a company description manually (optional):")
        st.markdown("### 💰 Transaction Size Filter")
        min_value = st.number_input("Minimum Enterprise Value (mln $)", min_value=0.0, value=0.0, step=10.0)
        max_value = st.number_input("Maximum Enterprise Value (mln $)", min_value=0.0, value=10_000.0, step=10.0)
        manual_industry = st.sidebar.selectbox(
        "🏷️ Filter by Industry (optional override):",
        options=["Detect Automatically"] + industry_list
        )
        start_search = st.button("🔍 Find Matches")
        if st.button("🔄 Restart"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

    if not start_search and st.session_state.get("generate_new", True):
        st.info("Enter a company website and/or description, then click **Find Matches** to start.")
        return

    query_text = ""

    if start_search or st.session_state.get("generate_new", False):
        if query_input and is_valid_url(query_input):
            with st.spinner("Scraping website..."):
                query_text = scrape_website(query_input)
                if not query_text:
                    st.warning("Website content could not be scraped.")

    if manual_description:
        query_text = manual_description.strip() + "\n" + query_text

    if not query_text.strip():
        st.info("Please enter a valid website or a company description to proceed.")
        return

    if st.session_state.generate_new:
        with st.spinner("Analyzing profile..."):
            try:
                df, industry_list = load_database()
                if manual_industry != "Detect Automatically":
                    pattern = re.escape(f"({manual_industry})")
                    df = df[df["Primary Industry"].str.contains(pattern, case=False, na=False)]
                    detected_industry = manual_industry
                else:
                    detected_industry = detect_industry_from_text(query_text, client)
                    if detected_industry:
                        df = df[df["Primary Industry"].str.contains(detected_industry, case=False, na=False)]
                    else:
                        st.warning("Could not detect industry. Showing all companies.")

                st.sidebar.markdown(f"**🧠 Industry Used:** {detected_industry or 'Not detected'}")

                df = df[
                    (df["Total Enterprise Value (mln$)"].fillna(0.0) >= min_value) &
                    (df["Total Enterprise Value (mln$)"].fillna(float('inf')) <= max_value)
                ]

                if df.empty:
                    st.warning("No companies match filters. Showing full list.")
                    df, industry_list = load_database()

                descriptions = df["Business Description"].astype(str).tolist()
                query_variants = [query_text] + paraphrase_query(query_text, client)
                if not query_variants:
                    st.error("Failed to generate query variants.")
                    st.stop()

                embeds = embed_text_batch(descriptions + query_variants, client)
                db_embeds = np.array(embeds[:len(descriptions)])
                query_embeds = np.array(embeds[len(descriptions):])
                if query_embeds.size == 0:
                    st.error("Query embedding is empty.")
                    st.stop()

                query_embed = np.mean(query_embeds, axis=0).reshape(1, -1)

                scores = cosine_similarity(db_embeds, query_embed).flatten()
                top_indices = get_top_indices(scores, SIMILARITY_THRESHOLD)[:40]
                df_top40 = df.iloc[top_indices].copy()

                with ThreadPoolExecutor() as executor:
                    scraped_texts = list(executor.map(scrape_website, df_top40["Web page"]))

                df_top40["Explanation"] = [explain_match(query_text, desc, client) for desc in df_top40["Business Description"]]
                full_texts = [f"{desc}\n{text}" for desc, text in zip(df_top40["Business Description"], scraped_texts)]
                final_embeds = np.array(embed_text_batch(full_texts + [query_text], client)[:-1])
                final_scores = cosine_similarity(final_embeds, query_embed).flatten()

                df_top40["Score"] = final_scores
                df_top40["ID"] = df_top40["MI Transaction ID"].astype(str)
                df_filtered = df_top40[~df_top40["ID"].isin(st.session_state.previous_matches)]
                df_final = df_filtered.nlargest(25, "Score")

                st.session_state.previous_matches.update(df_final["ID"].tolist())
                st.session_state.results = df_final
                st.session_state.generate_new = False

            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.stop()

    if st.session_state.results is not None:
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

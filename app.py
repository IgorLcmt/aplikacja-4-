import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import io
import re
import time

# ===== CONSTANTS =====
MAX_TEXT_LENGTH = 4000
BATCH_SIZE = 100
MAX_TOKENS = 8000
RATE_LIMIT_DELAY = 1.0  # Seconds between scrapes
SIMILARITY_THRESHOLD = 0.75

# ===== STREAMLIT CONFIG =====
st.set_page_config(page_title="CMT Company Analyzer 🔍", layout="wide")

# ===== INITIALIZATION =====
@st.cache_resource
def init_openai(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

# ===== DATA LOADING =====
@st.cache_data
def load_database() -> pd.DataFrame:
    try:
        df = pd.read_excel("app_data/Database.xlsx", engine="openpyxl")
        df.columns = [col.strip() for col in df.columns]
        required_cols = [
            'Target/Issuer Name', 'MI Transaction ID', 
            'Implied Enterprise Value/ EBITDA (x)', 'Business Description',
            'Primary Industry', 'Web page'
        ]
        return df.dropna(subset=required_cols)
    except Exception as e:
        st.error(f"Database loading failed: {str(e)}")
        st.stop()

# ===== EMBEDDING FUNCTIONS =====
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
            batch = clean_texts[i:i+BATCH_SIZE]
            response = _client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            embeddings.extend([record.embedding for record in response.data])
    except Exception as e:
        st.error(f"Embedding failed: {str(e)}")
        st.stop()
    
    return embeddings

# ===== GPT UTILITIES =====
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
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"GPT call failed: {str(e)}")
        return ""

def summarize_website(text: str, client: OpenAI) -> str:
    return gpt_chat("Summarize this web content in 3-5 concise bullet points.", text, client)

def explain_match(query: str, company_desc: str, client: OpenAI) -> str:
    return gpt_chat("You are a business analyst.",
                    f"Explain in 3-5 bullet points how this company matches the provided query.\\n\\nQuery:\\n{query}\\n\\nCompany:\\n{company_desc}",
                    client)

def generate_tags(description: str, client: OpenAI) -> str:
    return gpt_chat("Extract 3-5 high-level tags or categories from the business description.", description, client)

def paraphrase_query(query: str, client: OpenAI) -> List[str]:
    response = gpt_chat("Paraphrase this business query into 3 alternate versions.", query, client)
    return [line.strip("-• ") for line in response.splitlines() if line.strip()]

# ===== WEB SCRAPING =====
def is_valid_url(url: str) -> bool:
    return re.match(r'^https?://', url) is not None

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

# ===== CORE LOGIC =====
def get_top_indices(scores: np.ndarray, threshold: float) -> np.ndarray:
    qualified = scores >= threshold
    return np.argsort(-scores[qualified])

# ===== UI & SESSION STATE =====
def main():
    api_key = st.secrets["openai"]["api_key"]
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

    with st.sidebar:
        query_input = st.text_input("🌐 Paste company website URL:")

        if is_valid_url(query_input):
            with st.spinner("Scraping website..."):
                query_text = scrape_website(query_input)
                if not query_text:
                    st.error("Website content could not be scraped.")
                    return
        else:
            st.error("Please enter a valid website URL (http/https).")
            return
    
    if not query_input:
        st.info("Enter a company profile to begin")
        return

    if st.session_state.generate_new:
        with st.spinner("Analyzing profile..."):
            try:
                df = load_database()
                descriptions = df["Business Description"].astype(str).tolist()
                
                query_variants = [query_input] + paraphrase_query(query_input, client)
                embeds = embed_text_batch(descriptions + query_variants, client)
                db_embeds = np.array(embeds[:len(descriptions)])
                query_embeds = np.array(embeds[len(descriptions):])
                query_embed = np.mean(query_embeds, axis=0).reshape(1, -1)

                scores = cosine_similarity(db_embeds, query_embed).flatten()
                top_indices = get_top_indices(scores, SIMILARITY_THRESHOLD)[:20]
                df_top20 = df.iloc[top_indices].copy()

                with ThreadPoolExecutor() as executor:
                    scraped_texts = list(executor.map(scrape_website, df_top20["Web page"]))
                
                summaries = [summarize_website(text, client) for text in scraped_texts]
                explanations = [explain_match(query_input, desc, client) for desc in df_top20["Business Description"]]
                tags = [generate_tags(desc, client) for desc in df_top20["Business Description"]]

                df_top20["Summary"] = summaries
                df_top20["Explanation"] = explanations
                df_top20["Tags"] = tags

                full_texts = [f"{desc}\\n{text}" for desc, text in zip(df_top20["Business Description"], scraped_texts)]
                final_embeds = np.array(embed_text_batch(full_texts + [query_input], client)[:-1])
                final_scores = cosine_similarity(final_embeds, query_embed).flatten()

                df_top20["Score"] = final_scores
                df_top20["ID"] = df_top20["MI Transaction ID"].astype(str)
                df_filtered = df_top20[~df_top20["ID"].isin(st.session_state.previous_matches)]
                df_final = df_filtered.nlargest(10, "Score")

                st.session_state.previous_matches.update(df_final["ID"].tolist())
                st.session_state.results = df_final
                st.session_state.generate_new = False

            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.stop()

    if st.dataframe(st.session_state.results[[
        "Target/Issuer Name",
        "MI Transaction ID",
        "Implied Enterprise Value/ EBITDA (x)",
        "Company Geography (Target/Issuer)",  # If this column exists
        "Business Description",
        "Primary Industry",
        "Web page",
        "Score",
        "ID",
        "Explanation"  # GPT-generated
    ]], use_container_width=True)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            st.session_state.results.to_excel(writer, index=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download Excel",
                data=output.getvalue(),
                file_name="Company_Matches.xlsx"
            )
        with col2:
            if st.button("🔄 Find New Matches"):
                st.session_state.generate_new = True
                st.rerun()

if __name__ == "__main__":
    main()


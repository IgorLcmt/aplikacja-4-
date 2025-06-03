
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
     # client is used but not passed or defined here
    response = client.embeddings.create(...)
    clean_texts = [truncate_text(t.strip()) for t in texts if isinstance(t, str)]
    embeddings = embed_text_batch(texts, client)
    
    try:
        for i in range(0, len(clean_texts), BATCH_SIZE):
            batch = clean_texts[i:i+BATCH_SIZE]
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            embeddings.extend([record.embedding for record in response.data])
    except Exception as e:
        st.error(f"Embedding failed: {str(e)}")
        st.stop()
    
    return embeddings

# ===== WEB SCRAPING =====
def is_valid_url(url: str) -> bool:
    return re.match(r'^https?://', url) is not None

def scrape_website(url: str) -> str:
    if not is_valid_url(url):
        return ""
    
    try:
        time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:MAX_TEXT_LENGTH]
    except Exception:
        return ""

# ===== CORE LOGIC =====
def get_top_indices(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Returns indices of scores above threshold, sorted descending"""
    qualified = scores >= threshold
    return np.argsort(-scores[qualified])

# ===== UI & SESSION STATE =====
def main():
    # === Authentication ===
    api_key = st.secrets["openai"]["api_key"]
    if not api_key:
        st.error("OpenAI API key missing")
        st.stop()
    
       client = OpenAI(api_key=api_key)
    
    # === Session State ===
    session_defaults = {
        "results": None,
        "scraped_cache": {},
        "previous_matches": set(),
        "generate_new": True
    }
    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # === Sidebar Input ===
    with st.sidebar:
        query_input = st.text_area("📝 Paste company profile:", height=200)
    
    if not query_input:
        st.info("Enter a company profile to begin")
        return

    # === Processing Pipeline ===
    if st.session_state.generate_new:
        with st.spinner("Analyzing profile..."):
            try:
                # Load data
                df = load_database()
                descriptions = df["Business Description"].astype(str).tolist()
                
                # Initial embedding
                embeds = embed_text_batch(descriptions + [query_input], client)
                db_embeds = np.array(embeds[:-1])
                query_embed = np.array(embeds[-1]).reshape(1, -1)
                
                # Similarity calculation
                scores = cosine_similarity(db_embeds, query_embed).flatten()
                top_indices = get_top_indices(scores, SIMILARITY_THRESHOLD)[:20]
                df_top20 = df.iloc[top_indices].copy()

                # Parallel scraping
                with ThreadPoolExecutor() as executor:
                    scraped_texts = list(executor.map(scrape_website, df_top20["Web page"]))
                
                # Re-ranking
                full_texts = [f"{desc}\n{web}" for desc, web in 
                            zip(df_top20["Business Description"], scraped_texts)]
                final_embeds = np.array(embed_text_batch(full_texts + [query_input], client)[:-1])
                final_scores = cosine_similarity(final_embeds, query_embed).flatten()
                
                # Final selection
                df_top20["Score"] = final_scores
                df_top20["ID"] = df_top20["MI Transaction ID"].astype(str)
                df_filtered = df_top20[~df_top20["ID"].isin(st.session_state.previous_matches)]
                df_final = df_filtered.nlargest(10, "Score")
                
                # Update session
                st.session_state.previous_matches.update(df_final["ID"].tolist())
                st.session_state.results = df_final
                st.session_state.generate_new = False
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.stop()

    # === Display Results ===
    if st.session_state.results is not None:
        st.success("Top 10 Matches Found")
        st.dataframe(st.session_state.results, use_container_width=True)
        
        # Export controls
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

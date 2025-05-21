
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import os
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import requests
from bs4 import BeautifulSoup
import time
import joblib
import pickle
import io
import tiktoken

# === Streamlit Configuration ===
st.set_page_config(page_title="CMT analiza mnożników pod wycene 🔍", layout="wide")

# === Load API Key ===
api_key = st.secrets.get("openai", {}).get("api_key")
if not api_key:
    st.error("❌ OpenAI API key is missing. Please check your secrets configuration.")
    st.stop()

# === Initialize session state ===
if "results" not in st.session_state:
    st.session_state.results = None

if "scraped_cache" not in st.session_state:
    st.session_state.scraped_cache = {}

# === Load database ===
@st.cache_data
def load_database():
    df = pd.read_excel("app_data/Database.xlsx", engine="openpyxl")
    df.columns = [col.strip() for col in df.columns]
    df = df.rename(columns={
        'Business Description\n(Target/Issuer)': 'Business Description',
        'Primary Industry\n(Target/Issuer)': 'Primary Industry'
    })
    df = df.dropna(subset=[
        'Target/Issuer Name', 'MI Transaction ID', 'Implied Enterprise Value/ EBITDA (x)',
        'Business Description', 'Primary Industry'])
    return df

# === Embed text via OpenAI ===
from openai import OpenAI
import tiktoken

def embed_text_batch(texts, api_key, batch_size=100):
    client = OpenAI(api_key=api_key)
    clean_texts = [t.strip()[:4000] for t in texts if isinstance(t, str) and t.strip()]
    all_embeddings = []

    for i in range(0, len(clean_texts), batch_size):
        batch = clean_texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-ada-002"
        )
        all_embeddings.extend([record.embedding for record in response.data])

    return all_embeddings
# === Scrape web page ===
def scrape_website(url):
    if url in st.session_state.scraped_cache:
        return st.session_state.scraped_cache[url]
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        st.session_state.scraped_cache[url] = text
        return text
    except Exception as e:
        return ""

# === Sidebar input ===
query_input = st.sidebar.text_area("🎨 Paste company profile here:", height=200)

# === Main logic ===
if api_key and query_input and st.session_state.get("generate_new", True):
    # 🛡 Safety check
    if "embed_text_batch" not in globals():
        st.error("❌ Embedding function is not defined.")
        st.stop()
        
    with st.spinner("Embedding and finding initial matches..."):
        df = load_database()
        descriptions = df["Business Description"].dropna().astype(str).tolist()
        embeds = embed_text_batch(list(descriptions) + [query_input], api_key)
        db_embeds = np.array(embeds[:-1])
        query_embed = np.array(embeds[-1]).reshape(1, -1)
        scores = cosine_similarity(db_embeds, query_embed).flatten()
        top20_idx = np.argsort(scores)[-20:][::-1]
        df_top20 = df.iloc[top20_idx].copy()

    with st.spinner("Scraping top 20 sites..."):
        scraped_texts = []

        # 👉 Add this debug print here
        st.write("Available columns:", df_top20.columns.tolist())

        for url in df_top20["Web page"]:
            scraped_texts.append(scrape_website(url))

    # Step 1: Load previously selected matches (initialize if not present)
    if "previous_matches" not in st.session_state:
        st.session_state.previous_matches = set()

    # Step 2: Get final top 20 after re-ranking
    with st.spinner("Re-ranking after scraping..."):
        full_texts = [desc + "\n" + web for desc, web in zip(df_top20["Business Description"], scraped_texts)]
        embeds = embed_text_batch(full_texts + [query_input], api_key)
        final_embeds = np.array(embeds[:-1])
        final_query = np.array(embeds[-1]).reshape(1, -1)
        final_scores = cosine_similarity(final_embeds, final_query).flatten()

    df_top20["Score"] = final_scores

    # Step 3: Filter out previously shown matches
    df_top20["ID"] = df_top20["MI Transaction ID"].astype(str)  # or another unique field
    df_filtered = df_top20[~df_top20["ID"].isin(st.session_state.previous_matches)]

    # Step 4: Select 10 new matches
    df_final = df_filtered.sort_values("Score", ascending=False).head(10)

    # Step 5: Update session with selected IDs to avoid duplicates
    st.session_state.previous_matches.update(df_final["ID"].tolist())
    st.session_state.results = df_final

    # ✅ Reset flag after matches are generated
    st.session_state.generate_new = False
    
    st.success("🚀 Top 10 Matches Ready")
    st.dataframe(df_final, use_container_width=True)

    # === Excel export ===
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_final.to_excel(writer, index=False, sheet_name="Top Matches")
 
    # Show buttons only if results exist
if st.session_state.results is not None:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        st.session_state.results.to_excel(writer, index=False, sheet_name="Top Matches")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            "⬇️ Download Excel",
            data=output.getvalue(),
            file_name="Top_Matches.xlsx"
        )
    with col2:
        if st.button("🔄 Generate 10 New Matches"):
            st.session_state.generate_new = True
            st.experimental_rerun()  # Ensure immediate rerun

elif not query_input:
    st.info("👉 Enter a company profile to begin.")

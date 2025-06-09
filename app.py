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

# ===== INIT OPENAI =====
@st.cache_resource
def init_openai(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

# ===== LOAD DATABASE =====
@st.cache_data(show_spinner="Loading database...")
def load_database() -> tuple[pd.DataFrame, list]:
    try:
        df = pd.read_excel("app_data/Database.xlsx", engine="openpyxl")
        df.columns = [col.strip().replace('\xa0', ' ') for col in df.columns]

        val_col = "Total Enterprise Value (mln$)"
        if val_col in df.columns:
            df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

        industry_candidates = df["Primary Industry"].dropna().astype(str).tolist()

        cleaned = []
        for entry in industry_candidates:
            match = re.search(r"\((.*?)\)", entry)
            value = match.group(1).strip() if match else entry.strip()
            if value and any(c.isalpha() for c in value) and len(value) <= 157:
                cleaned.append(value)

        industry_list = sorted(set(cleaned))
        return df, industry_list

    except Exception as e:
        st.error(f"Database loading failed: {str(e)}")
        st.stop()

    st.write("All columns:", df.columns.tolist())

# ===== TEXT UTILS =====
def truncate_text(text: str, encoding_name: str = "cl100k_base") -> str:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)[:MAX_TOKENS]
    return encoding.decode(tokens)

@st.cache_data
def embed_text_batch(texts: List[str], _client: OpenAI) -> List[List[float]]:
    clean_texts = [truncate_text(t.strip()) for t in texts if isinstance(t, str)]
    embeddings = []
    for i in range(0, len(clean_texts), BATCH_SIZE):
        batch = clean_texts[i:i + BATCH_SIZE]
        response = _client.embeddings.create(input=batch, model="text-embedding-ada-002")
        embeddings.extend([record.embedding for record in response.data])
    return embeddings

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
    except Exception:
        return ""

def explain_match_structured(query: str, company_desc: str, similarity_score: float, client: OpenAI) -> str:
    prompt = f"""
You are a critical business analyst. Your task is to evaluate whether a transaction is a good match based on four key dimensions.

Start by considering the SIMILARITY SCORE, which is a float between 0 and 1.
Be skeptical if the score is low (< 0.6). Avoid forcing a match — it's okay to say it's weak.

SIMILARITY SCORE: {similarity_score:.2f}

Now assess the following, using a YES/NO answer and one-line justification:

1. **Industry match**: Do the industries clearly overlap?
2. **Product/service similarity**: Are the products or services comparable in type or use?
3. **Sales channel alignment**: Do they sell in similar ways (e.g., B2B SaaS, e-commerce)?
4. **Customer segment alignment**: Do they target similar customer types (e.g., enterprises, consumers, healthcare)?

Respond in this format exactly:

---
**Similarity Score**: X.XX  
**Industry Match**: YES/NO – Reason  
**Product/Service Similarity**: YES/NO – Reason  
**Sales Channel Alignment**: YES/NO – Reason  
**Customer Segment Match**: YES/NO – Reason  
**Overall Verdict**: STRONG MATCH / MODERATE MATCH / WEAK MATCH – Short reason
---

Query Profile:
{query}

Company Description:
{company_desc}
    """

    return gpt_chat("You are a critical business analyst.", prompt, client)

def summarize_scraped_text(raw_text: str, client: OpenAI) -> str:
    return gpt_chat(
        "Summarize the following website content. Focus on identifying the company's industry, business model, Sales channels, core products or services, and main customer types. In description don't add name of the company.",
        raw_text, client
    )

@st.cache_data(show_spinner="Fetching and summarizing website...")
def scrape_website_cached(url: str) -> str:
    return scrape_website(url)

def get_summarized_website(url: str, client: OpenAI) -> str:
    scraped = scrape_website_cached(url)
    return summarize_scraped_text(scraped, client) if scraped else ""

def paraphrase_query(query: str, client: OpenAI) -> List[str]:
    response = gpt_chat("Paraphrase this business query into 3 alternate versions.", query, client)
    return [line.strip("-• ") for line in response.splitlines() if line.strip()]

def detect_industry_from_text(text: str, client: OpenAI) -> str:
    return gpt_chat(
        "Identify the company's primary industry. Respond with one word like IT, Healthcare, Retail, etc.",
        text, client
    )

def is_valid_url(url: str) -> bool:
    return isinstance(url, str) and re.match(r'^https?://', url) is not None

def scrape_website(url: str) -> str:
    if not is_valid_url(url):
        return ""
    try:
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
    api_key = st.secrets.get("openai", {}).get("api_key")
    if not api_key:
        st.error("Missing OpenAI API key.")
        st.stop()
    client = OpenAI(api_key=api_key)

    session_defaults = {"results": None, "generate_new": False}
    for k, v in session_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    df, industry_list = load_database()

    # Sidebar
    with st.sidebar:
        query_input = st.text_input("🌐 Company website (optional):")
        min_value = st.number_input("Min Enterprise Value ($M)", 0.0, value=0.0)
        max_value = st.number_input("Max Enterprise Value ($M)", 0.0, value=10000.0)
        manual_industries = st.multiselect("🏷️ Filter by industry (optional):", options=industry_list)
        use_detected_also = st.checkbox("Include detected industry in filtering", value=False)

        if st.button("🔍 Find Matches"):
            st.session_state.generate_new = True
            st.rerun()
        if st.button("🔄 Restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if not st.session_state.generate_new and st.session_state.results is None:
        st.info("Enter a company description or website and click **Find Matches**.")
        return

    if st.session_state.generate_new:
        query_text = ""
        if query_input and is_valid_url(query_input):
            with st.spinner("Scraping and summarizing website..."):
                summarized = get_summarized_website(query_input, client)
                if summarized:
                    if "edited_summary" not in st.session_state:
                        st.session_state.edited_summary = summarized
                    st.session_state.edited_summary = st.text_area(
                        "📝 Website Summary (you can edit it before matching):",
                        value=st.session_state.edited_summary,
                        height=250,
                        key="edited_summary_input"
                    )
                else:
                    st.warning("Could not extract usable content from the website.")
        
        # ⬇️ Build the final query_text AFTER user confirms
        description_confirmed = st.checkbox("✅ I confirm the company description above is correct")
        if not description_confirmed:
            st.warning("Please confirm the company description before proceeding.")
            return
    
        query_text = st.session_state.get("edited_summary_input", "").strip()

        if not query_text:
            st.error("Please confirm or edit the company description above.")
            return
        
        # Save it back to session state (optional, for consistency)
        st.session_state["edited_summary"] = query_text

        with st.spinner("Analyzing profile..."):
            from difflib import get_close_matches
            detected_industry = detect_industry_from_text(query_text, client)
            st.info(f"Detected primary industry: **{detected_industry}**")

            industry_embeddings_batch = embed_text_batch([detected_industry], client)
            if not industry_embeddings_batch:
                st.error("Industry embedding failed.")
                st.stop()
            industry_embeddings = industry_embeddings_batch[0]

            unique_industries = df["Primary Industry"].dropna().astype(str).unique().tolist()
            industry_to_embed = [re.search(r"\((.*?)\)", i).group(1) if re.search(r"\((.*?)\)", i) else i for i in unique_industries]
            industry_to_embed = [i.strip() for i in industry_to_embed]

            embedded_db_industries = embed_text_batch(industry_to_embed, client)
            industry_scores = cosine_similarity([industry_embeddings], embedded_db_industries).flatten()
            top_indices = np.where(industry_scores > 0.75)[0]
            matching_industries = [unique_industries[i] for i in top_indices]
            initial_filter = df["Primary Industry"].isin(matching_industries)

            fuzzy_matches = []
            if manual_industries:
                raw_industries = df["Primary Industry"].astype(str).tolist()
                for selected in manual_industries:
                    fuzzy_matches.extend(get_close_matches(selected, raw_industries, n=20, cutoff=0.6))
                manual_filter = df["Primary Industry"].isin(fuzzy_matches)
                combined_filter = manual_filter  # ❗️Only use manual selection
            else:
                combined_filter = initial_filter if use_detected_also else df["Primary Industry"] != ""  # fallback

            df = df[combined_filter].copy()
            df = df[
                (df["Total Enterprise Value (mln$)"].fillna(0.0) >= min_value) &
                (df["Total Enterprise Value (mln$)"].fillna(float("inf")) <= max_value)
            ]
            if df.empty:
                st.error("No companies match your filters.")
                return

            descriptions = df["Business Description"].astype(str).tolist()
            query_variants = [query_text] + paraphrase_query(query_text, client)
            query_embeds = np.array(embed_text_batch(query_variants, client))
            query_embed = np.mean(query_embeds, axis=0).reshape(1, -1)
            db_embeds = np.array(embed_text_batch(descriptions, client))
            scores = cosine_similarity(db_embeds, query_embed).flatten()
            top_indices = np.argsort(-scores)[:20]
            df_top = df.iloc[top_indices].copy()
            df_top["Similarity Score"] = scores[top_indices]

            explanations = [
                explain_match_structured(query_text, desc, score, client)
                for desc, score in zip(df_top["Business Description"], df_top["Similarity Score"])
            ]

            relevant_industries = set(matching_industries + fuzzy_matches) if manual_industries else set(matching_industries)

            # 🔼 Add this block here to adjust score based on industry match
            INDUSTRY_BOOST = 0.10  # You can tune this value
            df_top["Adjusted Score"] = df_top.apply(
                lambda row: row["Similarity Score"] + INDUSTRY_BOOST
                if row["Primary Industry"] in relevant_industries else row["Similarity Score"],
                axis=1
            )
            
            # ✅ Now sort based on adjusted score
            df_top = df_top.sort_values("Adjusted Score", ascending=False)
            df_top["Explanation"] = explanations
            def count_no_answers(explanation: str) -> int:
            return len(re.findall(r"\*\*.*?:\*\* NO", explanation, re.IGNORECASE))
            df_top["NO Count"] = df_top["Explanation"].apply(count_no_answers)

            # ⛔️ Hard filter (remove poor matches)
            df_top = df_top[df_top["NO Count"] <= 3]

            df_top["Match Verdict"] = df_top["NO Count"].apply(
                lambda x: "❌ Poor Match" if x >= 3 else "✅ Relevant"
            )
      
            if manual_industries or use_detected_also:
                valid_industries = set(matching_industries + fuzzy_matches)
                df_top = df_top[df_top["Primary Industry"].isin(valid_industries)].copy()
                if df_top.empty:
                    st.warning("No top matches aligned with selected industries. Try relaxing filters.")

            st.session_state.results = df_top
            st.session_state.generate_new = False

    if st.session_state.results is not None:
        st.subheader("Top Matching Transactions")
        st.dataframe(st.session_state.results, use_container_width=True)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            st.session_state.results.to_excel(writer, index=False)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download Excel", data=output.getvalue(), file_name="Company_Matches.xlsx")
        with col2:
            if st.button("🔄 Find New Matches"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.cache_data.clear()
                st.rerun()

if __name__ == "__main__":
    main()

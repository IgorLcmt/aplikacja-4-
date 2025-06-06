import pandas as pd
import re
from openai import OpenAI
from newspaper import Article
from typing import List
from streamlit.runtime.caching import cache_data

MAX_TEXT_LENGTH = 4000
BATCH_SIZE = 100

# ===== LOAD DATABASE =====
@cache_data
def load_database() -> pd.DataFrame:
    return pd.read_excel("Database.xlsx")

# ===== WEBSITE SCRAPING =====
def is_valid_url(url: str) -> bool:
    return isinstance(url, str) and re.match(r"^https?://", url) is not None

def scrape_website(url: str) -> str:
    if not is_valid_url(url):
        return ""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text[:MAX_TEXT_LENGTH]
    except Exception:
        return ""

# ===== SUMMARY =====
def summarize_scraped_text(raw_text: str, client: OpenAI) -> str:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": "Summarize the following website content. Focus on identifying the company’s industry, core products or services, and main customer types.\n" + raw_text
        }]
    ).choices[0].message.content.strip()

# ===== EMBEDDING =====
def embed_text_batch(texts: List[str], client: OpenAI) -> List[List[float]]:
    results = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [r.embedding for r in results.data]

# ===== INDUSTRY DETECTION =====
def detect_industry_from_text(text: str, client: OpenAI) -> str:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": "Identify the company's primary industry. Respond with one word like IT, Healthcare, Retail, etc.\n" + text
        }]
    ).choices[0].message.content.strip()

# ===== EXPLANATION =====
def explain_match(query: str, company_desc: str, client: OpenAI) -> str:
    prompt = f"""
You are a business analyst. Based on the profile below, explain in 3–5 bullet points why this company is a

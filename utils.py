import pandas as pd
import numpy as np
import re
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

def load_database():
    df = pd.read_excel("app_data/Database.xlsx")
    industry_list = df["Primary Industry"].dropna().unique().tolist()
    return df, industry_list

def embed_text_batch(texts, client: OpenAI):
    return np.random.rand(len(texts), 768)

def summarize_scraped_text(url: str, client: OpenAI):
    return f"Summary for {url}"

def detect_industry_from_text(text: str, client: OpenAI):
    return "Manufacturing"

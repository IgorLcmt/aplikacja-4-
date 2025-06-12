
import numpy as np
import faiss
import pickle
from typing import List
import tiktoken
from openai import OpenAI

VECTOR_DB_PATH = "app_data/vector_db.index"
VECTOR_MAPPING_PATH = "app_data/vector_mapping.pkl"
MAX_TOKENS = 8000
BATCH_SIZE = 100

def truncate_text(text: str, encoding_name: str = "cl100k_base") -> str:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)[:MAX_TOKENS]
    return encoding.decode(tokens)

def embed_text_batch(texts: List[str], _client: OpenAI) -> List[List[float]]:
    clean_texts = [truncate_text(t.strip()) for t in texts if isinstance(t, str)]
    for t in clean_texts:
        assert len(t) > 20, f"Text too short: {t}"
    embeddings = []
    for i in range(0, len(clean_texts), BATCH_SIZE):
        batch = clean_texts[i:i + BATCH_SIZE]
        response = _client.embeddings.create(input=batch, model="text-embedding-3-large")
        embeddings.extend([record.embedding for record in response.data])
    return embeddings

def build_or_load_vector_db(embeddings: List[List[float]], metadata: List[str]):
    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    with open(VECTOR_MAPPING_PATH, "wb") as f:
        pickle.dump(metadata, f)
    faiss.write_index(index, VECTOR_DB_PATH)

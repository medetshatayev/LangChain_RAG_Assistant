import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

def embed_text(text: str) -> list[float]:
    try:
        model = SentenceTransformer("ibm-granite/granite-embedding-125m-english")
        embedding = model.encode(text).tolist()
        return embedding
    except Exception as e:
        print(f"Error during embedding: {e}")
        return []

if __name__ == '__main__':
    query = "How are you?"
    embeddings = embed_text(query)
    if embeddings:
        print(f'Query: "{query}"')
        print(f"Embedding dimension: {len(embeddings)}")
        print(f"First 5 embedding values: {embeddings[:5]}")
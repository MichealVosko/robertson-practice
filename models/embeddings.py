from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

# Load MedEmbed model once (cached)
embedder = SentenceTransformer("abhinand/MedEmbed-large-v0.1")

def embed_texts(texts: List[str]) -> np.ndarray:
    """Return L2-normalized embeddings as numpy arrays."""
    embs = embedder.encode(
        texts,
        convert_to_numpy=True,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embs

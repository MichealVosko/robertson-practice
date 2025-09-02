from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from typing import List, Dict, Any
## This file only contains embedding logic, ICD store builder, and rerank function.

__all__ = ["embed_texts", "build_icd_embedding_store", "rerank_icd_candidates"]

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

def build_icd_embedding_store(mapping_df, embed_fn):
    """
    Precompute embeddings for ICD codes/descriptions.
    Returns: dict[str, np.ndarray]
    """
    icd_store = {}
    for _, row in mapping_df.iterrows():
        key = f"{row['ICD-10 Code']}: {row['ICD-10 Description']}"
        if key not in icd_store:
            icd_store[key] = embed_fn([key])[0]
    return icd_store

# ICD_STORE should be initialized in app.py and passed in if needed

def rerank_icd_candidates(note_text: str,
                          icd_candidates: List[Dict[str, str]],
                          icd_store: Dict[str, np.ndarray],
                          top_k: int = 5,
                          rerank_with_cross_encoder: bool = True) -> List[Dict[str, Any]]:
    """
    Re-rank ICD candidates:
      1. Fast filter with embeddings (MedEmbed bi-encoder + ICD_STORE lookup).
      2. Optional cross-encoder rerank for top-K.
    """
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    if not icd_candidates:
        return []

    normalized_icds = []
    for c in icd_candidates:
        if "icd" in c and "description" in c:
            normalized_icds.append({"icd": c["icd"], "description": c["description"]})
        else:
            icd, desc = list(c.items())[0]
            normalized_icds.append({"icd": icd, "description": desc})

    note_emb = embed_texts([note_text])[0]
    icd_texts = [f'{c["icd"]}: {c["description"]}' for c in normalized_icds]
    icd_embs = np.array([icd_store[txt] for txt in icd_texts])

    sims = note_emb @ icd_embs.T
    idxs = np.argsort(-sims)[: min(top_k * 3, len(normalized_icds))]

    preselected = [
        {
            "icd": normalized_icds[i]["icd"],
            "description": normalized_icds[i]["description"],
            "score": float(sims[i]),
        }
        for i in idxs
    ]

    if rerank_with_cross_encoder and preselected:
        pairs = [(note_text, f"{p['icd']}: {p['description']}") for p in preselected]
        cross_scores = cross_encoder.predict(pairs)
        for p, cs in zip(preselected, cross_scores):
            p["cross_score"] = float(cs)
        preselected = sorted(preselected, key=lambda x: -x["cross_score"])

    unique_ranked = {}
    for r in preselected[:top_k]:
        icd = r["icd"]
        if icd not in unique_ranked or r.get("cross_score", r["score"]) > unique_ranked[icd].get("cross_score", r["score"]):
            unique_ranked[icd] = r

    return sorted(unique_ranked.values(), key=lambda x: -(x.get("cross_score", x["score"])))


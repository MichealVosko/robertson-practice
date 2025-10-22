import numpy as np
from typing import List, Dict, Any
from models.embeddings import embed_texts
from models.llm import icd_structured_llm
from langchain_core.prompts import PromptTemplate

def get_icd_candidates(predicted_cpt_codes: list[str], cpt_mapping: dict) -> list[dict]:
    icd_candidates = []
    for cpt in predicted_cpt_codes:
        if cpt in cpt_mapping:
            for icd_entry in cpt_mapping[cpt]["applicable_icds"]:
                for icd, desc in icd_entry.items():
                    icd_candidates.append({"icd": icd, "description": desc})
    return icd_candidates

def rerank_icd_candidates(note_text: str,
                          icd_candidates: List[Dict[str, str]],
                          top_k: int = 5) -> List[Dict[str, Any]]:
    if not icd_candidates:
        return []
    note_emb = embed_texts([note_text])[0]
    icd_texts = [f'{c["icd"]}: {c["description"]}' for c in icd_candidates]
    icd_embs = embed_texts(icd_texts)
    sims = note_emb @ icd_embs.T
    idxs = np.argsort(-sims)[:top_k]
    ranked = [{"icd": icd_candidates[i]["icd"], "description": icd_candidates[i]["description"], "score": float(sims[i])} for i in idxs]
    return ranked

icd_selection_prompt = PromptTemplate(
    input_variables=["note", "cpts", "allowed_icds"],
    template=("You are a medical coding assistant. Choose ICD-10 ONLY from allowed list.\n"
              "Clinical note:\n{note}\n\nCPT context: {cpts}\n\n"
              "Allowed ICDs:\n{allowed_icds}\n\n"
              "Return JSON: {{\"ICD10\": [\"code1\",\"code2\"]}}")
)

def select_icds_for_note(note_text: str, predicted_cpts: List[str], icd_candidates: List[Dict[str, str]], top_k: int = 15):
    ranked = rerank_icd_candidates(note_text, icd_candidates, top_k)
    allowed_icds_block = "\n".join([f"- {r['icd']} — {r['description']} (score {r['score']:.3f})" for r in ranked])
    prompt_str = icd_selection_prompt.format(note=note_text, cpts=", ".join(predicted_cpts), allowed_icds=allowed_icds_block)
    final = icd_structured_llm.invoke(prompt_str)
    return {"ranked": ranked, "final": final.ICD10}

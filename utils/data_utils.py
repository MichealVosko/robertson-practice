import pandas as pd
import streamlit as st
from utils.cpt_utils import get_cpt_mapping
from models.embeddings import build_icd_embedding_store, embed_texts

@st.cache_data
def load_mappings(file_path="data/Expanded_CPT_to_ICD_mapping.xlsx"):
    df = pd.read_excel(file_path)
    mapping = get_cpt_mapping(df)
    return df, mapping

@st.cache_resource
def build_embeddings(df):
    return build_icd_embedding_store(df, embed_texts)

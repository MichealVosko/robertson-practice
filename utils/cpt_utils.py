import pandas as pd
from langchain.prompts import PromptTemplate
from models.llm import structured_llm

cpt_prediction_prompt = """You are a medical coding assistant.
Assign the correct CPT code(s) from the allowed list below, based on the clinical note.
Do not guess codes that are not in the list.

Allowed CPTs:
- 90791: Psychiatric diagnostic evaluation
- 90832: Psychotherapy, 30 minutes with patient
- 90837: Psychotherapy, 60 minutes with patient
- H0004: Behavioral health counseling and therapy, per 15 minutes
- 96130: Psychological testing evaluation services, first hour
- 96131: Psychological testing evaluation services, each additional hour

Examples:
Note: "Patient presented for initial psychiatric diagnostic interview..." → CPT: 90791
Note: "Session lasted 60 minutes, focused on psychotherapy..." → CPT: 90837
Note: "Behavioral therapy session lasted 15 minutes..." → CPT: H0004

Now classify the following clinical note and return the result in the specified JSON format:
{soap_note}

Return the result in this JSON format:
{{
  "CPT": [ "code1", "code2" ]
}}
"""

def predict_cpt_code(soap_note: str):
    prompt = PromptTemplate(
        input_variables=["soap_note"],
        template=cpt_prediction_prompt
    )
    soap_note_prompt = prompt.format(soap_note=soap_note)
    response = structured_llm.invoke(soap_note_prompt)
    return response.CPT

def get_cpt_mapping(df: pd.DataFrame):
    cpt_mapping = {}
    for _, row in df.iterrows():
        cpt = str(row["CPT"]).strip()
        icd = str(row["ICD-10 Code"]).strip()
        if cpt not in cpt_mapping:
            cpt_mapping[cpt] = {"description": row["CPT Description"], "applicable_icds": []}
        cpt_mapping[cpt]["applicable_icds"].append({icd: row["ICD-10 Description"]})
    return cpt_mapping

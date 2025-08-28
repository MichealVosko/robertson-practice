from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List, Annotated
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Structured output for CPT
class CPT_Output(BaseModel):
    CPT: List[Annotated[str, Field(min_length=5, max_length=5, description="CPT code descrbing the chart note")]]

structured_llm = llm.with_structured_output(CPT_Output)

# Structured output for ICD
class ICD_Output(BaseModel):
    ICD10: List[str] = Field(
        default_factory=list,
        description="Final ICD-10 codes (1–4 most relevant)"
    )

icd_structured_llm = llm.with_structured_output(ICD_Output)

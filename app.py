import streamlit as st
import pandas as pd
import io
import os
from utils.pdf_utils import load_pdf, deidentify_and_strip
from utils.cpt_utils import predict_cpt_code, get_cpt_mapping
from utils.icd_utils import get_icd_candidates, select_icds_for_note
from models.embeddings import (
    embed_texts,
    build_icd_embedding_store,
    rerank_icd_candidates,
)
from utils.validation_utils import check_note
from utils.phi_utils import get_phi


cpt_icd_mapping_df = pd.read_excel("data/Expanded_CPT_to_ICD_mapping.xlsx")
cpt_mapping = get_cpt_mapping(cpt_icd_mapping_df)

st.title("Robertson Practice")

uploaded_files = st.file_uploader(
    "Upload one or more SOAP notes (PDFs)", type="pdf", accept_multiple_files=True
)

# If files uploaded, process them
if uploaded_files:
    # Clear previous session if file list changes
    if st.session_state.get("last_files") != [f.name for f in uploaded_files]:
        st.session_state.pop("results_df", None)
        st.session_state.pop("last_files", None)

    if "results_df" not in st.session_state:
        results = []
        total_files = len(uploaded_files)
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            file_path = f"data/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            text = load_pdf(file_path)
            clean = deidentify_and_strip(text)
            phi_data = get_phi(text)
            predicted_cpts = predict_cpt_code(clean)
            icd_candidates = get_icd_candidates(predicted_cpts, cpt_mapping)
            ranked_icds = rerank_icd_candidates(
                clean,
                icd_candidates,
                build_icd_embedding_store(cpt_icd_mapping_df, embed_texts),
                top_k=5,
            )
            final_selection = select_icds_for_note(
                clean, predicted_cpts, icd_candidates
            )
            validation_result = check_note(clean, uploaded_file.name)

            comments_str = (
                f"Missing: {', '.join(validation_result['missing_sections'])}"
                if validation_result["missing_sections"]
                else ""
            )
            row = {
                "Date": phi_data.get("Date", ""),
                "Appointment Type": "Therapy Session",
                "Client Name": phi_data.get("Patient", ""),
                "DOB": phi_data.get("DOB", ""),
                "Service Code": phi_data.get("Service Code", ""),
                "Primary Diagnosis": ", ".join(phi_data.get("Diagnosis Codes", [])),
                "Service Description": ", ".join(
                    [
                        cpt_icd_mapping_df[cpt_icd_mapping_df["CPT"] == cpt][
                            "CPT Description"
                        ].iloc[0]
                        for cpt in predicted_cpts
                        if not cpt_icd_mapping_df[
                            cpt_icd_mapping_df["CPT"] == cpt
                        ].empty
                    ]
                )
                if predicted_cpts
                else "",
                "Clinician Name": phi_data.get("Clinician", ""),
                "POS": phi_data.get("POS", ""),
                "Modifier": phi_data.get("Modifier", ""),
                "Coding": f"{', '.join(predicted_cpts)}--{', '.join(final_selection['final'])}",
                "Note Status": "Pending",
                "Status": "On Hold",
                "Comments": comments_str,
            }

            results.append(row)

            progress = idx / total_files
            progress_bar.progress(progress)
            status_text.text(
                f"Processing file {idx} of {total_files}: {uploaded_file.name}"
            )

        headers = [
            "Date",
            "Appointment Type",
            "Client Name",
            "DOB",
            "Service Code",

            "Service Description",
            "Clinician Name",
            "POS",
            "Modifier",
            "Coding",
            "Note Status",
            "Status",
            "Comments",
            
        ]

        st.session_state.results_df = pd.DataFrame(results, columns=headers)
        st.session_state.last_files = [f.name for f in uploaded_files]

    # Use cached results
    results_df = st.session_state.results_df
    st.subheader("Results Summary")
    st.dataframe(results_df, use_container_width=True)

    # Custom filename
    default_filename = "robertson_coding_solved"
    custom_name = st.text_input("Rename Excel file:", value=default_filename)
    if not custom_name.endswith(".xlsx"):
        custom_name += ".xlsx"

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Results")

    if st.download_button(
        label="📥 Download Results as Excel",
        data=buffer.getvalue(),
        file_name=custom_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ):
        # ✅ Clear session state after download
        st.session_state.pop("results_df", None)
        st.session_state.pop("last_files", None)

        # ✅ Optionally delete files from data/ folder
        for f in os.listdir("data"):
            if f.endswith(".pdf"):
                try:
                    os.remove(os.path.join("data", f))
                except Exception:
                    pass

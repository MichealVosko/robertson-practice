import streamlit as st
import pandas as pd
import io
import os
from utils.data_utils import load_mappings, build_embeddings
from utils.file_processing import process_file


cpt_icd_mapping_df, cpt_mapping = load_mappings()
icd_embedding_store = build_embeddings(cpt_icd_mapping_df)

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
            row = process_file(
                uploaded_file, cpt_mapping, cpt_icd_mapping_df, icd_embedding_store
            )

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

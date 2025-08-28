import streamlit as st
import pandas as pd
import io
from utils.pdf_utils import load_pdf, deidentify_and_strip
from utils.cpt_utils import predict_cpt_code, get_cpt_mapping
from utils.icd_utils import get_icd_candidates, select_icds_for_note
from utils.validation_utils import check_note

# Load mapping file once
cpt_icd_mapping_df = pd.read_excel("data/Expanded_CPT_to_ICD_mapping.xlsx")
cpt_mapping = get_cpt_mapping(cpt_icd_mapping_df)

st.title("Robertson Practice")

uploaded_files = st.file_uploader(
    "Upload one or more SOAP notes (PDFs)", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_files = len(uploaded_files)
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        # Save file to data folder
        file_path = f"data/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load + clean
        text = load_pdf(file_path)
        clean = deidentify_and_strip(text)

        # Predict CPT
        predicted_cpts = predict_cpt_code(clean)

        # Predict ICD
        icd_candidates = get_icd_candidates(predicted_cpts, cpt_mapping)
        final_selection = select_icds_for_note(clean, predicted_cpts, icd_candidates)

        # Validate note
        check = check_note(clean, uploaded_file.name)

        # Save result
        results.append({
            "filename": uploaded_file.name,
            "cpts": ", ".join(predicted_cpts),
            "icds": ", ".join(final_selection["final"]),
            "status": check["status"],
            "missing_sections": ", ".join(check["missing_sections"]) if check["missing_sections"] else ""
        })

        # Update progress
        progress = idx / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processing file {idx} of {total_files}: {uploaded_file.name}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    st.subheader("Results Summary")
    st.dataframe(results_df, use_container_width=True)

    # Download button for Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Results")
    st.download_button(
        label="📥 Download Results as Excel",
        data=buffer.getvalue(),
        file_name="coding_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

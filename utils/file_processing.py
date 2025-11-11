from utils.pdf_utils import load_pdf, deidentify_and_strip
from utils.cpt_utils import predict_cpt_code, calculate_cpt_units
from utils.icd_utils import get_icd_candidates, select_icds_for_note
from models.embeddings import rerank_icd_candidates
from utils.validation_utils import check_note
from utils.phi_utils import get_phi


def process_file(uploaded_file, cpt_mapping, cpt_icd_mapping_df, icd_embedding_store):
    """Process a single PDF and return a result row dictionary."""
    file_path = f"data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    text = load_pdf(file_path)
    clean = deidentify_and_strip(text)
    phi_data = get_phi(text)

    # CPT prediction & fix
    predicted_cpts = predict_cpt_code(clean)
    if "90840" in predicted_cpts and "90839" not in predicted_cpts:
        predicted_cpts.remove("90840")

    # ICD ranking
    icd_candidates = get_icd_candidates(predicted_cpts, cpt_mapping)
    ranked_icds = rerank_icd_candidates(
        clean, icd_candidates, icd_embedding_store, top_k=5
    )
    final_selection = select_icds_for_note(clean, predicted_cpts, ranked_icds)

    # Note validation
    validation_result = check_note(clean, uploaded_file.name)
    comments_str = (
        f"Missing: {', '.join(validation_result['missing_sections'])}"
        if validation_result["missing_sections"]
        else ""
    )

    # Duration and CPT units
    duration_str = phi_data.get("Duration")
    cpt_with_units = calculate_cpt_units(predicted_cpts, duration_str)

    # Build row dictionary
    service_descriptions = [
        cpt_icd_mapping_df[cpt_icd_mapping_df["CPT"] == cpt]["CPT Description"].iloc[0]
        for cpt in predicted_cpts
        if not cpt_icd_mapping_df[cpt_icd_mapping_df["CPT"] == cpt].empty
    ]

    row = {
        "Date": phi_data.get("Date", ""),
        "Appointment Type": "Therapy Session",
        "Client Name": phi_data.get("Patient", ""),
        "DOB": phi_data.get("DOB", ""),
        "Service Code": phi_data.get("Service Code", ""),
        "Primary Diagnosis": ", ".join(phi_data.get("Diagnosis Codes", [])),
        "Service Description": ", ".join(service_descriptions)
        if service_descriptions
        else "",
        "Clinician Name": phi_data.get("Clinician", ""),
        "POS": phi_data.get("POS", ""),
        "Modifier": phi_data.get("Modifier", ""),
        "Coding": f"{', '.join(cpt_with_units)}--{', '.join(final_selection['final'])}",
        "Note Status": "Pending",
        "Status": "On Hold",
        "Comments": comments_str,
    }

    return row

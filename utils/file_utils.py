from utils.pdf_utils import load_pdf, deidentify_and_strip
from utils.cpt_utils import predict_cpt_code, calculate_cpt_units
from utils.validation_utils import check_note
from utils.phi_utils import get_phi
from utils.psych_eval_utils import extract_psych_eval_data


def process_file(uploaded_file, cpt_icd_mapping_df):
    file_path = f"data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    text = load_pdf(file_path)
    clean = deidentify_and_strip(text)
    phi_data = get_phi(text)
    service_code = phi_data.get("Service Code", "")

    # Psych evaluation CPTs
    psych_cpts = ["96130", "96131", "96138", "96139"]

    if service_code in psych_cpts:
        # Run psych evaluation logic
        psych_data = extract_psych_eval_data(text)
        units = psych_data["Follow up code Units"]

        # CPT description safely
        service_descriptions = []
        service_desc_df = cpt_icd_mapping_df[cpt_icd_mapping_df["CPT"] == service_code]
        if not service_desc_df.empty:
            service_descriptions.append(service_desc_df["CPT Description"].iloc[0])

        # ICD codes from chart
        diagnosis_codes = phi_data.get("Diagnosis Codes", [])
        unit_str = f"{units}X" if units > 0 else ""
        row_coding = f"{service_code}--{unit_str}--{', '.join(diagnosis_codes)}"

        comments_str = "Check portal for evaluation file"

    else:
        # Normal flow for other CPTs (908x etc.)
        predicted_cpts = predict_cpt_code(clean)
        if "90840" in predicted_cpts and "90839" not in predicted_cpts:
            predicted_cpts.remove("90840")

        # ICD extraction from chart
        diagnosis_codes = phi_data.get("Diagnosis Codes", [])

        # Note validation
        validation_result = check_note(clean, uploaded_file.name)
        comments_str = (
            f"Missing: {', '.join(validation_result['missing_sections'])}"
            if validation_result["missing_sections"]
            else ""
        )
        
        clinician_name = phi_data.get("Clinician", "") or ""
        medicaid_clinicians = ["Kayla", "Kaeli", "Virginia", "Courtney"]
        
        is_medicaid_clinician = any(
            name.lower() in clinician_name.lower() for name in medicaid_clinicians
        )
        if service_code == "90837" and is_medicaid_clinician:
            note = "Verify the H0004 with Medicaid guidelines."
            comments_str = f"{comments_str} | {note}" if comments_str else note

        # Duration & CPT units
        duration_str = phi_data.get("Duration")
        cpt_with_units = calculate_cpt_units(predicted_cpts, duration_str)

        # CPT descriptions safely
        service_descriptions = []
        if service_code:
            service_desc_df = cpt_icd_mapping_df[
                cpt_icd_mapping_df["CPT"] == service_code
            ]
            if not service_desc_df.empty:
                service_descriptions.append(service_desc_df["CPT Description"].iloc[0])

        # Build coding string using predicted CPT units and ICDs
        row_coding = f"{', '.join(cpt_with_units)}--{', '.join(diagnosis_codes)}"

    # Build row dictionary
    row = {
        "Date": phi_data.get("Date", ""),
        "Appointment Type": "Therapy Session",
        "Client Name": phi_data.get("Patient", ""),
        "DOB": phi_data.get("DOB", ""),
        "Service Code": service_code,
        "Primary Diagnosis": ", ".join(diagnosis_codes),
        "Service Description": ", ".join(service_descriptions)
        if service_descriptions
        else "",
        "Clinician Name": phi_data.get("Clinician", ""),
        "POS": phi_data.get("POS", ""),
        "Modifier": phi_data.get("Modifier", ""),
        "Coding": row_coding,
        "Note Status": "Pending",
        "Status": "On Hold",
        "Comments": comments_str,
    }

    return row

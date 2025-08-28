required_sections = ["Interventions Used", "Risk Assessment", "Current Mental Status"]

def check_note(note_text: str, filename: str):
    missing = [section for section in required_sections if section not in note_text]
    return {"filename": filename, "status": "RED" if missing else "OK", "missing_sections": missing}

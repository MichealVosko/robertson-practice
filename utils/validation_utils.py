required_sections = {
    "Interventions Used": r"\bInterventions\s+Used\b",
    # Match "Risk Assessment" OR "Assessment" optionally followed by
    # words, slashes, or spaces
    "Risk Assessment": r"\b(Risk\s+Assessment|Assessment(\s*[/\-\w\s]+)*)\b",
    "Current Mental Status": r"\bCurrent\s+Mental\s+Status\b"
}


def check_note(note_text: str, filename: str):
    missing = [section for section in required_sections if section not in note_text]
    return {"filename": filename, "status": "RED" if missing else "OK", "missing_sections": missing}

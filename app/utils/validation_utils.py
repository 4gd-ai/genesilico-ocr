def is_name_match(extracted_name: dict, existing_name: dict) -> bool:
    """
    Compare extracted and existing patient names.
    Returns True if both first and last names match (case-insensitive).
    """
    def normalize(name):
        return name.strip().lower() if name else ""

    return (
        normalize(extracted_name.get("firstName")) == normalize(existing_name.get("firstName")) and
        normalize(extracted_name.get("lastName")) == normalize(existing_name.get("lastName"))
    )


def check_name_conflict(
    extracted_data: dict,
    existing_data: dict,
    templates: dict
) -> str:
    """
    Checks if extracted patient name matches the existing name.
    Returns a formatted conflict message if mismatch is found.
    """
    extracted_name = extracted_data.get("patientInformation", {}).get("patientName", {})
    existing_name = existing_data.get("patientInformation", {}).get("patientName", {})

    if not is_name_match(extracted_name, existing_name):
        return templates["field_conflict"].format(
            field_path="patientInformation.patientName",
            ocr_value=extracted_name,
            expected_value=existing_name,
            reason="the extracted name does not match the existing patient record"
        )
    return ""
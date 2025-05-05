"""Knowledge base for the AI agent with information about the TRF schema."""

import json
from typing import Dict, List, Any

# TRF schema knowledge

TRF_SCHEMA_OVERVIEW = """
The Test Requisition Form (TRF) schema contains the following main sections:
- Patient Information: Basic details about the patient
- Clinical Summary: Medical diagnosis and related information
- Family History: Cancer history in the family
- Hospital Information: Details about the hospital
- Physician Information: Details about the treating physician
- Lab Information: Details about the laboratory
- Sample Information: Details about the collected samples
- Shipment Details: Information about sample shipment
- Consent: Patient consent information
"""

FIELD_DESCRIPTIONS = {
    "patientID": "Unique identifier for the patient",
    "reportId": "Report ID",
    "orderID": "Order ID",
    "gssampleID": "Genomics sample ID",

    "patientInformation.patientName.firstName": "Patient's first name",
    "patientInformation.patientName.middleName": "Patient's middle name",
    "patientInformation.patientName.lastName": "Patient's last name",
    "patientInformation.gender": "Patient's gender",
    "patientInformation.ethnicity": "Patient's ethnicity",
    "patientInformation.dob": "Date of birth (MM/DD/YYYY)",
    "patientInformation.age": "Age in years",
    "patientInformation.weight": "Weight",
    "patientInformation.height": "Height",
    "patientInformation.email": "Email",
    "patientInformation.patientInformationPhoneNumber": "Phone number",
    "patientInformation.patientInformationAddress": "Full address",
    "patientInformation.patientCountry": "Country",
    "patientInformation.patientState": "State",
    "patientInformation.patientCity": "City",
    "patientInformation.patientPincode": "Postal/ZIP code",

    "clinicalSummary.primaryDiagnosis": "Primary diagnosis",
    "clinicalSummary.initialDiagnosisStage": "Initial diagnosis stage",
    "clinicalSummary.currentDiagnosis": "Current diagnosis",
    "clinicalSummary.diagnosisDate": "Date of diagnosis",
    "clinicalSummary.Immunohistochemistry.er": "Estrogen Receptor (ER) status (e.g., positive, negative, or percentage)",
    "clinicalSummary.Immunohistochemistry.pr": "Progesterone Receptor (PR) status (e.g., positive, negative, or percentage)",
    "clinicalSummary.Immunohistochemistry.her2neu": "HER2/neu status (e.g., positive, negative, or score)",
    "clinicalSummary.Immunohistochemistry.ki67": "Ki-67 index (percentage)",
    "clinicalSummary.pastIllness": "Relevant past illness or medical history",
    "clinicalSummary.comments": "Additional clinical notes such as prior tests or observations",

    "FamilyHistory.familyHistoryOfAnyCancer": "Family history of cancer",
    "FamilyHistory.familyMember": "Family members with cancer history (required if history is 'Yes')",

    "physician.physicianName": "Physician's name",
    "physician.physicianSpecialty": "Physician's specialty",
    "physician.physicianPhoneNumber": "Physician's phone number",
    "physician.physicianEmail": "Physician's email",

    "hospital.hospitalName": "Hospital name",
    "hospital.hospitalAddress": "Hospital address",

    # Fixed Sample field descriptions to reflect array structure
    "Sample.sampleCollectionType": "How the sample was collected (e.g., Core Needle Biopsy)",
    "Sample.sampleType": "Type of the biological sample (e.g., blood, tissue)",
    "Sample.sampleID": "Unique identifier for the sample",
    "Sample.sampleCollectionSite": "Sites from which the sample was collected (e.g., 'Left Breast, Upper Outer Quadrant', Right Breast)",
    "Sample.sampleCollectionDate": "Date the sample was collected",
    "Sample.timeOfCollection": "Time the sample was collected",
    "Sample.currentStatusOfSample": "Array of current condition/status of the sample",
    "Sample.selectTheTemperatureAtWhichItIsStored": "Storage temperature",
    "Sample.sampleUHID": "Sample-specific UHID",
    "Sample.storedIn": "Array of storage locations"
}

REQUIRED_FIELDS_INFO = [
    "patientID",
    "patientInformation.patientName.firstName",
    "patientInformation.patientName.lastName",
    "patientInformation.gender",
    "patientInformation.dob",
    "patientInformation.patientInformationPhoneNumber",
    "clinicalSummary.primaryDiagnosis",
]

COMMON_EXTRACTION_ERRORS = [
    "Name split errors between first/middle/last",
    "Inconsistent date formats (e.g., MM/DD/YYYY vs. DD/MM/YYYY)",
    "Gender may appear as M/F or full word",
    "Phone formats vary with spaces, dashes, or brackets",
    "Addresses often multi-line or unordered",
    "Diagnosis may include stage info mixed in",
    "IHC markers like ER/PR/HER2 may appear as '7+', '2-', or similar, which must be mapped to 'positive' or 'negative'",
    "Sample fields may be extracted as single values instead of arrays where arrays are required"
]

EXTRACTION_HEURISTICS = [
    "Use label separators like colons to find values",
    "Patient names are usually near the top",
    "Normalize all dates to MM/DD/YYYY",
    "Map IHC scores: any value ending with '+' (e.g., '2+', '3+', '7+') → 'positive', any value ending with '-' → 'negative'",
    "Detect sample type keywords (blood, tissue, marrow) to map sample field",
    "Use regex to capture and interpret IHC expressions like ER: 3+ or HER2: 1-",
    "Match fields by semantic similarity if label or section names differ but intent is equivalent",
    "Ensure Sample is always formatted as an array of objects, even when only one sample exists",
    "Convert string values to arrays for fields expecting arrays (sampleCollectionSite, currentStatusOfSample, storedIn)"
]

FIELD_RELATIONSHIPS = [
    "If 'PatientReport.FamilyHistory.familyHistoryOfAnyCancer' is 'Yes', then 'PatientReport.FamilyHistory.familyMember' must contain entries with relationToPatient, typesOfCancer, and EstimatedAgeAtOnset",
    "If 'PatientReport.clinicalSummary.Immunohistochemistry.hasPatientFailedPriorTreatment' is 'Yes', then 'pastTherapy' should be filled",
    "Extracted 'patientInformation.patientName.firstName' and 'lastName' must match the existing record for verification."
]

FIELD_ALIASES = {
    "Specimen Type": "Sample.sampleType",
    "Specimen Site": "Sample.sampleCollectionSite",
    "Collection Date": "Sample.sampleCollectionDate",
    "Specimen ID": "Sample.sampleID",
    "ER status": "clinicalSummary.Immunohistochemistry.er",
    "PR status": "clinicalSummary.Immunohistochemistry.pr",
    "HER2 score": "clinicalSummary.Immunohistochemistry.her2neu",
    "Ki67": "clinicalSummary.Immunohistochemistry.ki67",
    "Patient Phone No.": "patientInformation.patientInformationPhoneNumber",
    "Diagnosis Date": "clinicalSummary.diagnosisDate",
    "History": "clinicalSummary.pastIllness",
    "Previous Tests": "clinicalSummary.comments"
}

NORMALIZATION_RULES = {
    "Immunohistochemistry": {
        "positive_markers": ["1+", "2+", "3+", "4+", "5+", "6+", "7+"],
        "negative_markers": ["0", "0-", "1-", "2-", "3-", "4-", "5-", "6-", "7-"]
    }
}

FIELD_CONSTRAINTS = {
    "patientInformation.age": {"type": "number", "min": 0, "max": 120},
    "clinicalSummary.Immunohistochemistry.ki67": {"type": "number", "min": 0, "max": 100},
    "patientInformation.email": {"type": "string", "format": "email"},
    "patientInformation.patientInformationPhoneNumber": {"type": "string", "pattern": "^\\d{10}$"},
    "patientInformation.dob": {"type": "string", "format": "date"}
}

# Structure validation rules
STRUCTURE_VALIDATION = {
    "array_fields": [
        "Sample",
        "Sample.sampleCollectionSite",
        "Sample.currentStatusOfSample", 
        "Sample.storedIn",
        "clinicalSummary.Immunohistochemistry.diseaseStatusAtTheTimeOfTesting",
        "clinicalSummary.Immunohistochemistry.pastTherapy",
        "clinicalSummary.Immunohistochemistry.currentTherapy",
        "FamilyHistory.familyMember"
    ]
}

# Agent response templates
FIELD_SUGGESTION_TEMPLATE = """
I've analyzed the document and have a suggestion for the field '{field_path}':

Current value: {current_value}
Suggested value: {suggested_value}
Confidence: {confidence:.0%}

Reason: {reason}

Would you like to accept this suggestion?
"""

MISSING_FIELDS_TEMPLATE = """
I've identified the following required fields that are missing:

{missing_fields_list}

Would you like me to suggest values for any of these fields based on the document?
"""

FIELD_CONFLICT_TEMPLATE = """
I've detected a potential conflict in the extracted data:

Field: {field_path}
OCR extracted value: {ocr_value}
Expected value based on other fields: {expected_value}

This discrepancy might be due to {reason}.

Which value would you prefer to use?
"""

STRUCTURE_ERROR_TEMPLATE = """
I've detected an issue with the structure of the data:

Field: {field_path}
Expected structure: {expected_structure}
Actual structure: {actual_structure}

I'll automatically fix this by converting it to the proper format.
"""

# Export for easy access by the agent
KNOWLEDGE_BASE = {
    "schema_overview": TRF_SCHEMA_OVERVIEW,
    "field_descriptions": FIELD_DESCRIPTIONS,
    "required_fields": REQUIRED_FIELDS_INFO,
    "common_extraction_errors": COMMON_EXTRACTION_ERRORS,
    "extraction_heuristics": EXTRACTION_HEURISTICS,
    "field_relationships": FIELD_RELATIONSHIPS,
    "field_aliases": FIELD_ALIASES,
    "normalization_rules": NORMALIZATION_RULES,
    "field_constraints": FIELD_CONSTRAINTS,
    "structure_validation": STRUCTURE_VALIDATION,
    "templates": {
        "field_suggestion": FIELD_SUGGESTION_TEMPLATE,
        "missing_fields": MISSING_FIELDS_TEMPLATE,
        "field_conflict": FIELD_CONFLICT_TEMPLATE,
        "structure_error": STRUCTURE_ERROR_TEMPLATE
    },
}

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
    "patientID": "A unique identifier for the patient",
    "reportId": "A unique identifier for the report",
    "limsID": "Laboratory Information Management System ID",
    "orderID": "A unique identifier for the order",
    
    "patientInformation.patientName.firstName": "Patient's first name",
    "patientInformation.patientName.middleName": "Patient's middle name (optional)",
    "patientInformation.patientName.lastName": "Patient's last name",
    "patientInformation.gender": "Patient's gender (Male, Female, or Other)",
    "patientInformation.ethnicity": "Patient's ethnicity",
    "patientInformation.dob": "Patient's date of birth (MM/DD/YYYY)",
    "patientInformation.age": "Patient's age in years",
    "patientInformation.weight": "Patient's weight",
    "patientInformation.height": "Patient's height",
    "patientInformation.email": "Patient's email address",
    "patientInformation.patientInformationPhoneNumber": "Patient's phone number",
    "patientInformation.patientInformationAddress": "Patient's complete address",
    "patientInformation.patientCountry": "Patient's country of residence",
    "patientInformation.patientState": "Patient's state/province of residence",
    "patientInformation.patientCity": "Patient's city of residence",
    "patientInformation.patientPincode": "Patient's postal/ZIP code",
    
    "clinicalSummary.primaryDiagnosis": "Patient's primary diagnosis",
    "clinicalSummary.initialDiagnosisStage": "Initial stage of diagnosis",
    "clinicalSummary.currentDiagnosis": "Current diagnosis",
    "clinicalSummary.diagnosisDate": "Date of diagnosis (MM/DD/YYYY)",
    "clinicalSummary.Immunohistochemistry.er": "Estrogen Receptor status (positive, negative, or percentage)",
    "clinicalSummary.Immunohistochemistry.pr": "Progesterone Receptor status (positive, negative, or percentage)",
    "clinicalSummary.Immunohistochemistry.her2neu": "HER2/neu status (positive, negative, or score)",
    "clinicalSummary.Immunohistochemistry.ki67": "Ki-67 index (percentage)",
    
    "FamilyHistory.familyHistoryOfAnyCancer": "Whether the patient has a family history of cancer (Yes/No)",
    
    "physician.physicianName": "Name of the treating physician",
    "physician.physicianSpecialty": "Specialty of the physician",
    "physician.physicianPhoneNumber": "Physician's phone number",
    "physician.physicianEmail": "Physician's email address",
    
    "hospital.hospitalName": "Name of the hospital",
    "hospital.hospitalAddress": "Address of the hospital",
    
    "Sample.0.sampleType": "Type of the biological sample (e.g., Blood, Tissue, Bone Marrow)",
    "Sample.0.sampleID": "Unique identifier for the sample",
    "Sample.0.sampleCollectionDate": "Date when the sample was collected (MM/DD/YYYY)",
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
    "Patient name may be split incorrectly between first, middle, and last names",
    "Date formats can be inconsistent (MM/DD/YYYY vs. DD/MM/YYYY vs. YYYY-MM-DD)",
    "Gender may be abbreviated (M/F) or spelled out (Male/Female)",
    "Phone numbers may include various formats with different separators",
    "Addresses may span multiple lines and include inconsistent formatting",
    "Diagnosis may include both the condition and stage information together",
]

EXTRACTION_HEURISTICS = [
    "Look for labels followed by a colon or similar separator to identify field values",
    "Patient names are typically near the top of the form",
    "Dates should be normalized to a consistent format (preferably MM/DD/YYYY)",
    "For immunohistochemistry results, look for +/- symbols or percentages",
    "Sample types are often indicated with checkboxes or circles",
]

FIELD_RELATIONSHIPS = [
    "If 'FamilyHistory.familyHistoryOfAnyCancer' is 'Yes', then 'FamilyHistory.familyMember' should be filled",
    "If 'clinicalSummary.Immunohistochemistry.hasPatientFailedPriorTreatment' is 'Yes', then 'clinicalSummary.Immunohistochemistry.pastTherapy' should be filled",
]

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

# Export for easy access by the agent
KNOWLEDGE_BASE = {
    "schema_overview": TRF_SCHEMA_OVERVIEW,
    "field_descriptions": FIELD_DESCRIPTIONS,
    "required_fields": REQUIRED_FIELDS_INFO,
    "common_extraction_errors": COMMON_EXTRACTION_ERRORS,
    "extraction_heuristics": EXTRACTION_HEURISTICS,
    "field_relationships": FIELD_RELATIONSHIPS,
    "templates": {
        "field_suggestion": FIELD_SUGGESTION_TEMPLATE,
        "missing_fields": MISSING_FIELDS_TEMPLATE,
        "field_conflict": FIELD_CONFLICT_TEMPLATE
    }
}

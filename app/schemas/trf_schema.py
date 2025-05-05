from typing import Dict, List, Set, Any, Tuple

# Define required fields in the TRF schema
REQUIRED_FIELDS = {
    "patientID",
    "patientInformation.patientName.firstName",
    "patientInformation.patientName.lastName",
    "patientInformation.gender",
    "patientInformation.dob",
    "patientInformation.patientInformationPhoneNumber",
    "clinicalSummary.primaryDiagnosis",
}

# Define field relationships and validations
FIELD_RELATIONSHIPS = [
    {
        "if_field": "clinicalSummary.Immunohistochemistry.hasPatientFailedPriorTreatment",
        "equals_value": "Yes",
        "then_require": ["clinicalSummary.Immunohistochemistry.pastTherapy"]
    },
    {
        "if_field": "FamilyHistory.familyHistoryOfAnyCancer",
        "equals_value": "Yes",
        "then_require": ["FamilyHistory.familyMember"]
    }
]

# Define field extraction patterns for commonly occurring fields
FIELD_EXTRACTION_PATTERNS = {
    "patientInformation.patientName.firstName": [
        r"(?:First\s+Name|Given\s+Name|Patient\s+First\s+Name)\s*:\s*([A-Za-z\s\-']+)",
        r"Name\s*:\s*([A-Za-z\-']+)\s+([A-Za-z\-']+)",
        r"Patient\s*:\s*([A-Za-z\-']+)\s+([A-Za-z\-']+)",
        r"Patient\s+Name\s*:\s*(?:Mrs\.|Mr\.|Ms\.|Dr\.|)?\s*([A-Za-z\-']+)"
    ],
    "patientInformation.patientName.lastName": [
        r"(?:Last\s+Name|Family\s+Name|Surname|Patient\s+Last\s+Name)\s*:\s*([A-Za-z\s\-']+)",
        r"Name\s*:\s*(?:[A-Za-z\-']+)\s+([A-Za-z\-']+)",
        r"Patient\s+Name\s*:\s*(?:Mrs\.|Mr\.|Ms\.|Dr\.|)?\s*[A-Za-z\-']+\s+([A-Za-z\s\-']+)"
    ],
    "patientInformation.gender": [
        r"(?:Gender|Sex)\s*:\s*(Male|Female|Other|M|F|m|f)",
        r"(?:Gender|Sex)\s*[:\)]\s*[☐|☑|☒|✓|✔]\s*(Male|Female|Other|M|F)",
        r"Age/Gender\s*:\s*\d+\s*Years?/(M|F)"
    ],
    "patientInformation.dob": [
        r"(?:DOB|Date\s+of\s+Birth|Birth\s+Date)\s*:\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
        r"(?:DOB|Date\s+of\s+Birth|Birth\s+Date)\s*:\s*(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})"
    ],
    "patientInformation.age": [
        r"(?:Age|Patient\s+Age)\s*:\s*(\d{1,3})\s*(?:years|yrs|yr|y)?",
        r"Age/Gender\s*:\s*(\d+)\s*Years?",
        r"(\d{1,3})\s*(?:years|yrs|yr|y)\s*(?:old)?"
    ],
    "patientInformation.mrnUhid": [
        r"(?:MRN|UHID|Medical\s+Record\s+Number|Hospital\s+ID|Patient\s+ID)\s*:\s*([A-Za-z0-9\-]+)",
        r"(?:MRN|UHID)\s*[#]?\s*([A-Za-z0-9\-]+)",
        r"UHID/MR\s+No\s*:\s*([A-Za-z0-9\-]+)"
    ],
    "patientInformation.patientInformationPhoneNumber": [
        r"(?:Phone|Tel|Telephone|Contact|Mobile|Cell|Phone\s+Number)\s*:\s*(\+?[0-9\-\(\)\s\.]{7,})",
        r"(?:Phone|Tel|Telephone|Contact|Mobile|Cell)\s*[#]?\s*(\+?[0-9\-\(\)\s\.]{7,})"
    ],
    "patientInformation.email": [
        r"(?:Email|E-mail|Electronic\s+Mail)\s*:\s*([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})",
        r"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})"
    ],
    "clinicalSummary.primaryDiagnosis": [
        r"(?:Diagnosis|Primary\s+Diagnosis|Clinical\s+Diagnosis|Provisional\s+Diagnosis)\s*:\s*([^\n\r]+)",
        r"(?:Diagnosis|Primary\s+Diagnosis|Clinical\s+Diagnosis)\s*[:\)]\s*([^\n\r]+)"
    ],
    "clinicalSummary.diagnosisDate": [
        r"(?:Diagnosis\s+Date|Date\s+of\s+Diagnosis)\s*:\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
        r"(?:Diagnosis\s+Date|Date\s+of\s+Diagnosis)\s*:\s*(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})"
    ],
    "clinicalSummary.Immunohistochemistry.er": [
        r"(?:ER|Estrogen\s+Receptor)\s*:\s*(\+|\-|Positive|Negative|Pos|Neg|[0-9]+%)",
        r"(?:ER|Estrogen\s+Receptor)\s*[:\)]\s*[☐|☑|☒|✓|✔]\s*(\+|\-|Positive|Negative|Pos|Neg)"
    ],
    "clinicalSummary.Immunohistochemistry.pr": [
        r"(?:PR|Progesterone\s+Receptor)\s*:\s*(\+|\-|Positive|Negative|Pos|Neg|[0-9]+%)",
        r"(?:PR|Progesterone\s+Receptor)\s*[:\)]\s*[☐|☑|☒|✓|✔]\s*(\+|\-|Positive|Negative|Pos|Neg)"
    ],
    "clinicalSummary.Immunohistochemistry.her2neu": [
        r"(?:HER2|HER2\/neu|Her-2\/neu)\s*:\s*(\+|\+\+|\+\+\+|\-|Positive|Negative|Pos|Neg|[0-9]+\+|[0-9])",
        r"(?:HER2|HER2\/neu|Her-2\/neu)\s*[:\)]\s*[☐|☑|☒|✓|✔]\s*(\+|\+\+|\+\+\+|\-|Positive|Negative|Pos|Neg)"
    ],
    "clinicalSummary.Immunohistochemistry.ki67": [
        r"(?:Ki-?67|Ki67)\s*:\s*([0-9]+%|[0-9]+)",
        r"(?:Ki-?67|Ki67)\s*[:\)]\s*([0-9]+%|[0-9]+)"
    ],
    "physician.physicianName": [
        r"(?:Doctor|Dr\.|Physician|Oncologist|Treating\s+Doctor|Referring\s+Doctor|Attending\s+Physician|Ref\s+Doctor)\s*:\s*([A-Za-z\s\.\-']+)",
        r"(?:Doctor|Dr\.|Physician|Oncologist)\s*[:\)]\s*([A-Za-z\s\.\-']+)"
    ],
    "physician.physicianEmail": [
        r"(?:Doctor|Physician|Oncologist|Provider)(?:'s)?\s+Email\s*:\s*([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})",
        r"(?:Doctor|Physician|Oncologist|Provider)\s+(?:[A-Za-z\s\.\-']+)\s+Email\s*:\s*([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})"
    ],
    "hospital.hospitalName": [
        r"(?:Hospital|Facility|Center|Medical\s+Center|Clinic|Institution)\s*:\s*([A-Za-z\s\.\-'&,]+)",
        r"(?:Hospital|Facility|Center|Medical\s+Center|Clinic)\s*[:\)]\s*([A-Za-z\s\.\-'&,]+)"
    ],
    "Sample.0.sampleType": [
        r"(?:Sample\s+Type|Specimen\s+Type|Type\s+of\s+Sample|Type\s+of\s+Specimen)\s*:\s*([^\n\r:]+)",
        r"(?:Sample|Specimen)\s*[:\)]\s*[☐|☑|☒|✓|✔]\s*(Blood|Tissue|Bone\s+Marrow|Swab|Saliva|Urine|CSF|Plasma|Serum)"
    ],
    "Sample.0.sampleID": [
        r"(?:Sample\s+ID|Specimen\s+ID|Sample\s+Number|Specimen\s+Number|Case\s+Id)\s*:\s*([A-Za-z0-9\-/]+)",
        r"(?:Sample|Specimen)\s+(?:ID|Number)\s*[#]?\s*([A-Za-z0-9\-/]+)"
    ],
    "Sample.0.sampleCollectionDate": [
        r"(?:Collection\s+Date|Date\s+of\s+Collection|Sample\s+Collection\s+Date|Specimen\s+Collection\s+Date|Collected)\s*:\s*(\d{1,2}\s*[/\-\.]\s*\w+[/\-\.]\d{2,4})",
        r"(?:Collection\s+Date|Date\s+of\s+Collection|Collected)\s*:\s*(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})"
    ],
}

# Helper methods for schema validation
def get_field_value(data: Dict[str, Any], field_path: str) -> Any:
    """Get value from nested dictionary using dot notation with array support"""
    if not field_path:
        return None
        
    parts = field_path.split('.')
    current = data
    
    for part in parts:
        if current is None:
            return None
            
        # Handle array indices in field path (e.g., "Sample.0.sampleType")
        if part.isdigit() and isinstance(current, list):
            index = int(part)
            if index < len(current):
                current = current[index]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
            
    return current

def set_field_value(data: Dict[str, Any], field_path: str, value: Any) -> None:
    """Set a value in a nested dictionary using a dot-separated path."""
    if not field_path or not data or not isinstance(data, dict):
        return
    
    parts = field_path.split('.')
    current = data
    
    # Navigate to the parent object, creating objects as needed
    for i, part in enumerate(parts[:-1]):
        # Handle array indices
        if '[' in part and ']' in part:
            array_name = part.split('[')[0]
            index_str = part.split('[')[1].split(']')[0]
            
            if array_name not in current:
                current[array_name] = []
                
            try:
                index = int(index_str)
                # Ensure list is long enough
                while len(current[array_name]) <= index:
                    current[array_name].append({})
                
                # Access the array element
                if not isinstance(current[array_name][index], dict):
                    current[array_name][index] = {}
                current = current[array_name][index]
            except (ValueError, IndexError, TypeError):
                # Skip this part if we can't parse the index
                return
        else:
            # Create new object if it doesn't exist
            if part not in current:
                current[part] = {}
            # Make sure we have a dict, not some other type
            if not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
    
    # Set the value on the parent
    last_part = parts[-1]
    if '[' in last_part and ']' in last_part:
        array_name = last_part.split('[')[0]
        index_str = last_part.split('[')[1].split(']')[0]
        
        if array_name not in current:
            current[array_name] = []
            
        try:
            index = int(index_str)
            # Ensure list is long enough
            while len(current[array_name]) <= index:
                current[array_name].append(None)
            current[array_name][index] = value
        except (ValueError, IndexError, TypeError):
            # Skip if we can't parse the index
            return
    else:
        # Just set the value
        current[last_part] = value

def validate_trf_data(trf_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate TRF data against the schema.
    
    Args:
        trf_data: The TRF data to validate
        
    Returns:
        Tuple of (is_valid, missing_required_fields, validation_errors)
    """
    missing_required_fields = []
    validation_errors = []
    
    # Check required fields
    for field in REQUIRED_FIELDS:
        value = get_field_value(trf_data, field)
        if value is None or value == "":
            missing_required_fields.append(field)
    
    # Check field relationships
    for relationship in FIELD_RELATIONSHIPS:
        field_value = get_field_value(trf_data, relationship["if_field"])
        if field_value == relationship["equals_value"]:
            for required_field in relationship["then_require"]:
                if get_field_value(trf_data, required_field) is None:
                    validation_errors.append(f"Field '{required_field}' is required when '{relationship['if_field']}' equals '{relationship['equals_value']}'")
    
    is_valid = len(missing_required_fields) == 0 and len(validation_errors) == 0
    return is_valid, missing_required_fields, validation_errors

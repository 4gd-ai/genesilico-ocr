import re
import time
from typing import Dict, List, Any, Tuple, Optional
import json
import asyncio

from ..models.document import OCRResult
from ..models.trf import PatientReport
from ..schemas.trf_schema import FIELD_EXTRACTION_PATTERNS, get_field_value, set_field_value


class FieldExtractor:
    """Extract fields from OCR results based on TRF schema."""
    
    def __init__(self, ocr_result: OCRResult):
        """Initialize the field extractor with OCR results."""
        self.ocr_result = ocr_result
        self.extracted_data = {}
        self.confidence_scores = {}
        self.extraction_stats = {
            "total_fields": 0,
            "extracted_fields": 0,
            "high_confidence_fields": 0,
            "low_confidence_fields": 0
        }
    
    async def extract_fields(self) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
        """
        Extract fields from OCR text using pattern matching and heuristics.
        
        Returns:
            Tuple of (extracted_data, confidence_scores, extraction_stats)
        """
        start_time = time.time()
        
        # Initialize a new PatientReport with default values
        trf_data = {"patientID": f"TEMP-{int(time.time())}"}
        
        # Get the full OCR text
        ocr_text = self.ocr_result.text
        
        # Track extracted fields and their confidence
        self.extracted_data = {}
        self.confidence_scores = {}
        
        # Apply pattern-based extraction
        for field_path, patterns in FIELD_EXTRACTION_PATTERNS.items():
            # Try each pattern until one matches
            for pattern in patterns:
                match = re.search(pattern, ocr_text, re.IGNORECASE)
                if match:
                    # Extract the value
                    value = match.group(1).strip()
                    
                    # Special case handling
                    if field_path == "patientInformation.gender":
                        # Normalize gender values
                        gender_map = {
                            "m": "Male", "male": "Male", "man": "Male",
                            "f": "Female", "female": "Female", "woman": "Female"
                        }
                        value = gender_map.get(value.lower(), value)
                    
                    # Compute confidence based on pattern match quality and OCR confidence
                    # For simplicity, we're using a fixed confidence for pattern matches
                    # In a production system, this would be more sophisticated
                    confidence = 0.8  # Base confidence for pattern matches
                    
                    # Store the extracted value and confidence
                    self.extracted_data[field_path] = value
                    self.confidence_scores[field_path] = confidence
                    
                    # Set the value in the TRF data
                    try:
                        set_field_value(trf_data, field_path, value)
                    except Exception as e:
                        print(f"Error setting field {field_path}: {e}")
                    
                    # Break once we've found a match
                    break
        
        # Update extraction statistics
        self.extraction_stats["total_fields"] = len(FIELD_EXTRACTION_PATTERNS)
        self.extraction_stats["extracted_fields"] = len(self.extracted_data)
        self.extraction_stats["high_confidence_fields"] = sum(1 for conf in self.confidence_scores.values() if conf >= 0.7)
        self.extraction_stats["low_confidence_fields"] = sum(1 for conf in self.confidence_scores.values() if conf < 0.7)
        self.extraction_stats["extraction_time"] = time.time() - start_time
        
        return trf_data, self.confidence_scores, self.extraction_stats
    
    async def extract_patient_info(self, trf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract patient information fields.
        
        Args:
            trf_data: Existing TRF data dictionary
            
        Returns:
            Updated TRF data dictionary
        """
        ocr_text = self.ocr_result.text
        
        # Initialize patient information if not exists
        if "patientInformation" not in trf_data or not isinstance(trf_data["patientInformation"], dict):
            trf_data["patientInformation"] = {}
        
        # Initialize patient name if not exists
        if "patientName" not in trf_data["patientInformation"] or not isinstance(trf_data["patientInformation"]["patientName"], dict):
            trf_data["patientInformation"]["patientName"] = {}
        
        # Extract patient name (first, middle, last)
        # First try the full name pattern
        full_name_pattern = r"(?:Patient\s+Name|Name|Patient)\s*:\s*(?:Mrs\.|Mr\.|Ms\.|Dr\.|)?\s*([A-Za-z\s\-']+)"
        full_name_match = re.search(full_name_pattern, ocr_text, re.IGNORECASE)
        
        if full_name_match:
            full_name = full_name_match.group(1).strip()
            name_parts = full_name.split()
            
            if len(name_parts) >= 2:
                # Assume first and last name
                trf_data["patientInformation"]["patientName"]["firstName"] = name_parts[0]
                self.confidence_scores["patientInformation.patientName.firstName"] = 0.8
                
                if len(name_parts) >= 3:
                    # Assume first, middle, last name
                    trf_data["patientInformation"]["patientName"]["middleName"] = name_parts[1]
                    trf_data["patientInformation"]["patientName"]["lastName"] = " ".join(name_parts[2:])
                    self.confidence_scores["patientInformation.patientName.middleName"] = 0.7
                    self.confidence_scores["patientInformation.patientName.lastName"] = 0.8
                else:
                    # Assume first and last name only
                    trf_data["patientInformation"]["patientName"]["lastName"] = name_parts[1]
                    self.confidence_scores["patientInformation.patientName.lastName"] = 0.8
        
        # Extract contact information
        phone_pattern = r"(?:Phone|Tel|Telephone|Contact|Mobile|Cell|Phone\s+Number)\s*:\s*(\+?[0-9\-\(\)\s\.]{7,})"
        phone_match = re.search(phone_pattern, ocr_text, re.IGNORECASE)
        if phone_match:
            trf_data["patientInformation"]["patientInformationPhoneNumber"] = phone_match.group(1).strip()
            self.confidence_scores["patientInformation.patientInformationPhoneNumber"] = 0.8
        
        # Extract address information
        address_pattern = r"(?:Address|Patient\s+Address)\s*:\s*([^\n\r]+)"
        address_match = re.search(address_pattern, ocr_text, re.IGNORECASE)
        if address_match:
            trf_data["patientInformation"]["patientInformationAddress"] = address_match.group(1).strip()
            self.confidence_scores["patientInformation.patientInformationAddress"] = 0.7
            
            # Try to parse city, state, zip from address
            city_state_zip_pattern = r"([A-Za-z\s]+),\s*([A-Za-z\s]+),?\s*(\d{5}(?:-\d{4})?)"
            csz_match = re.search(city_state_zip_pattern, address_match.group(1), re.IGNORECASE)
            if csz_match:
                trf_data["patientInformation"]["patientCity"] = csz_match.group(1).strip()
                trf_data["patientInformation"]["patientState"] = csz_match.group(2).strip()
                trf_data["patientInformation"]["patientPincode"] = csz_match.group(3).strip()
                self.confidence_scores["patientInformation.patientCity"] = 0.7
                self.confidence_scores["patientInformation.patientState"] = 0.7
                self.confidence_scores["patientInformation.patientPincode"] = 0.8
        
        return trf_data
    
    async def extract_clinical_summary(self, trf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract clinical summary information.
        
        Args:
            trf_data: Existing TRF data dictionary
            
        Returns:
            Updated TRF data dictionary
        """
        ocr_text = self.ocr_result.text
        
        # Initialize clinical summary if not exists
        if "clinicalSummary" not in trf_data or not isinstance(trf_data["clinicalSummary"], dict):
            trf_data["clinicalSummary"] = {}
        
        # Extract primary diagnosis
        diagnosis_pattern = r"(?:Diagnosis|Primary\s+Diagnosis|Clinical\s+Diagnosis|Provisional\s+Diagnosis)\s*:\s*([^\n\r]+)"
        diagnosis_match = re.search(diagnosis_pattern, ocr_text, re.IGNORECASE)
        if diagnosis_match:
            trf_data["clinicalSummary"]["primaryDiagnosis"] = diagnosis_match.group(1).strip()
            self.confidence_scores["clinicalSummary.primaryDiagnosis"] = 0.8
        
        # Extract diagnosis date
        date_pattern = r"(?:Diagnosis\s+Date|Date\s+of\s+Diagnosis)\s*:\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})"
        date_match = re.search(date_pattern, ocr_text, re.IGNORECASE)
        if date_match:
            trf_data["clinicalSummary"]["diagnosisDate"] = date_match.group(1).strip()
            self.confidence_scores["clinicalSummary.diagnosisDate"] = 0.8
        
        # Initialize Immunohistochemistry if not exists
        if "Immunohistochemistry" not in trf_data["clinicalSummary"] or not isinstance(trf_data["clinicalSummary"]["Immunohistochemistry"], dict):
            trf_data["clinicalSummary"]["Immunohistochemistry"] = {}
        
        # Look for immunohistochemistry report section
        if "IMMUNOHISTOCHEMISTRY REPORT" in ocr_text.upper():
            # Extract ER status
            er_pattern = r"(?:ER|Estrogen\s+Receptor)\s*:\s*(\+|\-|Positive|Negative|Pos|Neg|[0-9]+\s*%)"
            er_match = re.search(er_pattern, ocr_text, re.IGNORECASE)
            if er_match:
                trf_data["clinicalSummary"]["Immunohistochemistry"]["er"] = er_match.group(1).strip()
                self.confidence_scores["clinicalSummary.Immunohistochemistry.er"] = 0.8
            
            # Extract PR status
            pr_pattern = r"(?:PR|Progesterone\s+Receptor)\s*:\s*(\+|\-|Positive|Negative|Pos|Neg|[0-9]+\s*%)"
            pr_match = re.search(pr_pattern, ocr_text, re.IGNORECASE)
            if pr_match:
                trf_data["clinicalSummary"]["Immunohistochemistry"]["pr"] = pr_match.group(1).strip()
                self.confidence_scores["clinicalSummary.Immunohistochemistry.pr"] = 0.8
            
            # Extract HER2 status
            her2_pattern = r"(?:HER2|HER2\/neu|Her-2\/neu)\s*:\s*(\+|\+\+|\+\+\+|\-|Positive|Negative|Pos|Neg|[0-9]+\+|[0-9])"
            her2_match = re.search(her2_pattern, ocr_text, re.IGNORECASE)
            if her2_match:
                trf_data["clinicalSummary"]["Immunohistochemistry"]["her2neu"] = her2_match.group(1).strip()
                self.confidence_scores["clinicalSummary.Immunohistochemistry.her2neu"] = 0.8
            
            # Extract Ki67 status
            ki67_pattern = r"(?:Ki-?67|Ki67)\s*:\s*([0-9]+%|[0-9]+)"
            ki67_match = re.search(ki67_pattern, ocr_text, re.IGNORECASE)
            if ki67_match:
                trf_data["clinicalSummary"]["Immunohistochemistry"]["ki67"] = ki67_match.group(1).strip()
                self.confidence_scores["clinicalSummary.Immunohistochemistry.ki67"] = 0.8
        
        return trf_data
    
    async def extract_physician_info(self, trf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract physician information.
        
        Args:
            trf_data: Existing TRF data dictionary
            
        Returns:
            Updated TRF data dictionary
        """
        ocr_text = self.ocr_result.text
        
        # Initialize physician if not exists
        if "physician" not in trf_data or not isinstance(trf_data["physician"], dict):
            trf_data["physician"] = {}
        
        # Extract physician name
        physician_pattern = r"(?:Doctor|Dr\.|Physician|Oncologist|Treating\s+Doctor|Referring\s+Doctor|Attending\s+Physician|Ref\s+Doctor)\s*:\s*([A-Za-z\s\.\-']+)"
        physician_match = re.search(physician_pattern, ocr_text, re.IGNORECASE)
        if physician_match:
            trf_data["physician"]["physicianName"] = physician_match.group(1).strip()
            self.confidence_scores["physician.physicianName"] = 0.8
        
        # Extract physician email
        email_pattern = r"(?:Doctor|Physician|Oncologist|Provider)(?:'s)?\s+Email\s*:\s*([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})"
        email_match = re.search(email_pattern, ocr_text, re.IGNORECASE)
        if email_match:
            trf_data["physician"]["physicianEmail"] = email_match.group(1).strip()
            self.confidence_scores["physician.physicianEmail"] = 0.9
        
        # Extract physician phone
        phone_pattern = r"(?:Doctor|Physician|Oncologist|Provider)(?:'s)?\s+(?:Phone|Tel|Telephone|Contact)\s*:\s*(\+?[0-9\-\(\)\s\.]{7,})"
        phone_match = re.search(phone_pattern, ocr_text, re.IGNORECASE)
        if phone_match:
            trf_data["physician"]["physicianPhoneNumber"] = phone_match.group(1).strip()
            self.confidence_scores["physician.physicianPhoneNumber"] = 0.8
        
        # Extract physician specialty
        specialty_pattern = r"(?:Specialty|Specialization|Speciality)\s*:\s*([A-Za-z\s\.\-']+)"
        specialty_match = re.search(specialty_pattern, ocr_text, re.IGNORECASE)
        if specialty_match:
            trf_data["physician"]["physicianSpecialty"] = specialty_match.group(1).strip()
            self.confidence_scores["physician.physicianSpecialty"] = 0.8
        
        return trf_data
    
    async def extract_sample_info(self, trf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract sample information.
        
        Args:
            trf_data: Existing TRF data dictionary
            
        Returns:
            Updated TRF data dictionary
        """
        ocr_text = self.ocr_result.text
        
        # Initialize Sample array if not exists
        if "Sample" not in trf_data or not isinstance(trf_data["Sample"], list):
            trf_data["Sample"] = [{}]
        elif len(trf_data["Sample"]) == 0:
            trf_data["Sample"].append({})
        
        # Make sure we have at least one sample object
        if not isinstance(trf_data["Sample"][0], dict):
            trf_data["Sample"][0] = {}
        
        # Extract sample type
        sample_type_pattern = r"(?:Sample\s+Type|Specimen\s+Type|Type\s+of\s+Sample|Type\s+of\s+Specimen)\s*:\s*([^\n\r:]+)"
        sample_type_match = re.search(sample_type_pattern, ocr_text, re.IGNORECASE)
        if sample_type_match:
            trf_data["Sample"][0]["sampleType"] = sample_type_match.group(1).strip()
            self.confidence_scores["Sample.0.sampleType"] = 0.8
        
        # Extract sample ID
        sample_id_pattern = r"(?:Sample\s+ID|Specimen\s+ID|Sample\s+Number|Specimen\s+Number|Case\s+Id)\s*:\s*([A-Za-z0-9\-/]+)"
        sample_id_match = re.search(sample_id_pattern, ocr_text, re.IGNORECASE)
        if sample_id_match:
            trf_data["Sample"][0]["sampleID"] = sample_id_match.group(1).strip()
            self.confidence_scores["Sample.0.sampleID"] = 0.9
        
        # Extract sample collection date
        collection_date_pattern = r"(?:Collection\s+Date|Date\s+of\s+Collection|Sample\s+Collection\s+Date|Specimen\s+Collection\s+Date|Collected)\s*:\s*(\d{1,2}\s*[/\-\.]\s*\w+[/\-\.]\d{2,4})"
        collection_date_match = re.search(collection_date_pattern, ocr_text, re.IGNORECASE)
        if collection_date_match:
            trf_data["Sample"][0]["sampleCollectionDate"] = collection_date_match.group(1).strip()
            self.confidence_scores["Sample.0.sampleCollectionDate"] = 0.8
        
        # Extract storage temperature
        temp_pattern = r"(?:Storage\s+Temperature|Temperature|Stored\s+at)\s*:\s*((?:-|\+)?[0-9]+(?:\.[0-9]+)?\s*(?:°C|C|°F|F|K))"
        temp_match = re.search(temp_pattern, ocr_text, re.IGNORECASE)
        if temp_match:
            trf_data["Sample"][0]["selectTheTemperatureAtWhichItIsStored"] = temp_match.group(1).strip()
            self.confidence_scores["Sample.0.selectTheTemperatureAtWhichItIsStored"] = 0.7
        
        # Extract sample collection site
        site_pattern = r"(?:Collection\s+Site|Site\s+of\s+Collection|Sample\s+Collection\s+Site)\s*:\s*([^\n\r:]+)"
        site_match = re.search(site_pattern, ocr_text, re.IGNORECASE)
        if site_match:
            if not isinstance(trf_data["Sample"][0].get("sampleCollectionSite", None), list):
                trf_data["Sample"][0]["sampleCollectionSite"] = []
            trf_data["Sample"][0]["sampleCollectionSite"].append(site_match.group(1).strip())
            self.confidence_scores["Sample.0.sampleCollectionSite"] = 0.7
        
        return trf_data
    
    async def extract_hospital_info(self, trf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract hospital information.
        
        Args:
            trf_data: Existing TRF data dictionary
            
        Returns:
            Updated TRF data dictionary
        """
        ocr_text = self.ocr_result.text
        
        # Initialize hospital if not exists
        if "hospital" not in trf_data or not isinstance(trf_data["hospital"], dict):
            trf_data["hospital"] = {}
        
        # Extract hospital name
        hospital_pattern = r"(?:Hospital|Facility|Center|Medical\s+Center|Clinic|Institution|Client\s+Name)\s*:\s*([A-Za-z\s\.\-'&,]+)"
        hospital_match = re.search(hospital_pattern, ocr_text, re.IGNORECASE)
        if hospital_match:
            hospital_name = hospital_match.group(1).strip()
            if hospital_name and hospital_name != "-":
                trf_data["hospital"]["hospitalName"] = hospital_name
                self.confidence_scores["hospital.hospitalName"] = 0.8
        
        # Extract hospital address
        address_pattern = r"(?:Hospital|Facility|Institution)\s+Address\s*:\s*([^\n\r]+)"
        address_match = re.search(address_pattern, ocr_text, re.IGNORECASE)
        if address_match:
            trf_data["hospital"]["hospitalAddress"] = address_match.group(1).strip()
            self.confidence_scores["hospital.hospitalAddress"] = 0.7
        
        # Extract hospital contact person
        contact_person_pattern = r"(?:Contact\s+Person|Hospital\s+Contact)\s*:\s*([A-Za-z\s\.\-']+)"
        contact_person_match = re.search(contact_person_pattern, ocr_text, re.IGNORECASE)
        if contact_person_match:
            trf_data["hospital"]["contactPersonNameHospital"] = contact_person_match.group(1).strip()
            self.confidence_scores["hospital.contactPersonNameHospital"] = 0.8
        
        return trf_data
    
    def get_field_confidence(self, field_path: str) -> float:
        """Get the confidence score for a specific field."""
        return self.confidence_scores.get(field_path, 0.0)
    
    def get_low_confidence_fields(self, threshold: float = 0.7) -> List[str]:
        """Get a list of fields with confidence below the threshold."""
        return [field for field, conf in self.confidence_scores.items() if conf < threshold]
    
    def get_high_confidence_fields(self, threshold: float = 0.7) -> List[str]:
        """Get a list of fields with confidence above or equal to the threshold."""
        return [field for field, conf in self.confidence_scores.items() if conf >= threshold]

from typing import Dict, List, Any, Tuple, Set
import json

from ..schemas.trf_schema import REQUIRED_FIELDS, FIELD_RELATIONSHIPS, get_field_value, validate_trf_data


class SchemaValidator:
    """Validate TRF data against the schema."""
    
    @staticmethod
    def validate_trf_data(trf_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate TRF data against the schema.
        
        Args:
            trf_data: The TRF data to validate
            
        Returns:
            Tuple of (is_valid, missing_required_fields, validation_errors)
        """
        return validate_trf_data(trf_data)
    
    @staticmethod
    def get_completion_percentage(trf_data: Dict[str, Any]) -> float:
        """
        Calculate the percentage of required fields that are completed.
        
        Args:
            trf_data: The TRF data to check
            
        Returns:
            Completion percentage (0.0 to 1.0)
        """
        # Count how many required fields are filled
        filled_required_fields = 0
        for field in REQUIRED_FIELDS:
            value = get_field_value(trf_data, field)
            if value not in (None, "", []):
                filled_required_fields += 1
        
        # Calculate percentage
        completion_percentage = filled_required_fields / len(REQUIRED_FIELDS) if REQUIRED_FIELDS else 1.0
        return completion_percentage
    
    @staticmethod
    def get_missing_required_fields(trf_data: Dict[str, Any]) -> List[str]:
        """
        Get a list of required fields that are missing from the TRF data.
        
        Args:
            trf_data: The TRF data to check
            
        Returns:
            List of missing required field paths
        """
        missing_fields = []
        for field in REQUIRED_FIELDS:
            value = get_field_value(trf_data, field)
            if value in (None, "", []):
                missing_fields.append(field)
        
        return missing_fields
    
    @staticmethod
    def get_conditional_required_fields(trf_data: Dict[str, Any]) -> List[str]:
        """
        Get a list of conditionally required fields based on the current TRF data.
        
        Args:
            trf_data: The TRF data to check
            
        Returns:
            List of conditionally required field paths
        """
        conditional_fields = []
        for relationship in FIELD_RELATIONSHIPS:
            field_value = get_field_value(trf_data, relationship["if_field"])
            if field_value == relationship["equals_value"]:
                for required_field in relationship["then_require"]:
                    conditional_fields.append(required_field)
        
        return conditional_fields
    
    @staticmethod
    def validate_field_value(field_path: str, field_value: Any) -> Tuple[bool, str]:
        """
        Validate a specific field value.
        
        Args:
            field_path: Path to the field
            field_value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Handle specific field validations
        if field_path == "patientInformation.gender":
            valid_genders = {"Male", "Female", "Other", "M", "F"}
            if field_value and field_value not in valid_genders:
                return False, f"Invalid gender value: {field_value}. Expected one of: {', '.join(valid_genders)}"
        
        elif field_path == "patientInformation.email":
            if field_value and "@" not in field_value:
                return False, f"Invalid email format: {field_value}"
        
        elif field_path == "patientInformation.patientInformationPhoneNumber":
            if field_value and not any(c.isdigit() for c in field_value):
                return False, f"Phone number must contain digits: {field_value}"
        
        # Add more field-specific validations as needed
        
        # Default: field is valid
        return True, ""
    
    @staticmethod
    def generate_form_status(trf_data: Dict[str, Any]) -> str:
        """
        Generate a status for the TRF form based on validation results.
        
        Args:
            trf_data: The TRF data to check
            
        Returns:
            Form status string
        """
        is_valid, missing_required, validation_errors = validate_trf_data(trf_data)
        
        if not missing_required and not validation_errors:
            return "complete"
        elif len(missing_required) <= 3 and not validation_errors:
            return "nearly_complete"
        elif get_field_value(trf_data, "patientInformation"):
            return "partial"
        else:
            return "incomplete"

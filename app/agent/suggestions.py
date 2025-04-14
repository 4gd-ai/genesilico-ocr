"""AI agent suggestion generation for TRF data completion and correction."""

import time
from typing import Dict, List, Any, Tuple, Optional

from ..models.document import OCRResult
from ..schemas.trf_schema import REQUIRED_FIELDS, get_field_value
from .reasoning import agent_reasoning
from .knowledge_base import KNOWLEDGE_BASE


class AgentSuggestions:
    """Generate suggestions for TRF data completion and correction."""
    
    @staticmethod
    async def generate_suggestions(ocr_result: OCRResult, trf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate suggestions for completing and correcting TRF data.
        
        Args:
            ocr_result: OCR result with extracted text
            trf_data: Current TRF data
            
        Returns:
            Dictionary of suggestions
        """
        # Get analysis results from agent reasoning
        analysis_results = await agent_reasoning.analyze_ocr_result(ocr_result, trf_data)
        
        # Create suggestions response
        return {
            "document_id": ocr_result.document_id,
            "ocr_result_id": ocr_result.id,
            "suggestions": analysis_results["suggestions"],
            "missing_fields": analysis_results["missing_fields"],
            "low_confidence_fields": analysis_results["low_confidence_fields"],
            "completion_percentage": analysis_results["completion_percentage"],
            "timestamp": time.time()
        }
    
    @staticmethod
    async def get_field_suggestions(field_path: str, ocr_text: str, trf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get suggestions for a specific field.
        
        Args:
            field_path: Path to the field
            ocr_text: OCR extracted text
            trf_data: Current TRF data
            
        Returns:
            Field suggestion result
        """
        # Get suggestion from agent reasoning
        suggestion = await agent_reasoning.suggest_field_value(field_path, ocr_text, trf_data)
        
        # Format suggestion with field description
        field_description = KNOWLEDGE_BASE["field_descriptions"].get(field_path, f"Field {field_path}")
        suggestion["field_description"] = field_description
        
        # Add current value if it exists
        current_value = get_field_value(trf_data, field_path)
        suggestion["current_value"] = current_value
        
        return suggestion
    
    @staticmethod
    async def get_missing_field_suggestions(ocr_text: str, trf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get suggestions for all missing required fields.
        
        Args:
            ocr_text: OCR extracted text
            trf_data: Current TRF data
            
        Returns:
            Dictionary of missing field suggestions
        """
        # Identify missing required fields
        missing_fields = []
        for field in REQUIRED_FIELDS:
            value = get_field_value(trf_data, field)
            if value in (None, "", []):
                missing_fields.append(field)
        
        # Generate suggestions for each missing field
        suggestions = []
        for field_path in missing_fields:
            suggestion = await agent_reasoning.suggest_field_value(field_path, ocr_text, trf_data)
            if "error" not in suggestion and suggestion["suggested_value"]:
                field_description = KNOWLEDGE_BASE["field_descriptions"].get(field_path, f"Field {field_path}")
                suggestion["field_description"] = field_description
                suggestions.append(suggestion)
        
        return {
            "missing_fields": missing_fields,
            "suggestions": suggestions,
            "timestamp": time.time()
        }
    
    @staticmethod
    async def get_completion_guidance(trf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get guidance for completing the TRF form.
        
        Args:
            trf_data: Current TRF data
            
        Returns:
            Completion guidance
        """
        # Identify missing required fields
        missing_fields = []
        for field in REQUIRED_FIELDS:
            value = get_field_value(trf_data, field)
            if value in (None, "", []):
                field_description = KNOWLEDGE_BASE["field_descriptions"].get(field, f"Field {field}")
                missing_fields.append({
                    "field_path": field,
                    "description": field_description
                })
        
        # Calculate completion percentage
        completion_percentage = 0.0
        if REQUIRED_FIELDS:
            completed_count = len(REQUIRED_FIELDS) - len(missing_fields)
            completion_percentage = completed_count / len(REQUIRED_FIELDS)
        
        # Generate guidance message
        guidance_message = ""
        if not missing_fields:
            guidance_message = "All required fields are completed. The TRF is ready for submission."
        elif len(missing_fields) == 1:
            guidance_message = f"You're almost there! Just one required field left to complete: {missing_fields[0]['field_path']} ({missing_fields[0]['description']})."
        elif len(missing_fields) <= 3:
            guidance_message = f"You're making good progress! Just {len(missing_fields)} required fields left to complete."
        else:
            guidance_message = f"There are {len(missing_fields)} required fields that need to be completed."
        
        return {
            "completion_percentage": completion_percentage,
            "missing_fields": missing_fields,
            "guidance_message": guidance_message,
            "timestamp": time.time()
        }

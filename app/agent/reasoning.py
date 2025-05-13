"""Reasoning engine for the AI agent to analyze TRF data and make suggestions."""

import json
import time
import re
from typing import Dict, List, Any, Tuple, Optional
import asyncio
from ..utils.validation_utils import check_name_conflict, check_hospital_conflict

# Import Google Generative AI with error handling
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    # Fallback to a mock implementation for development/testing
    class GenAIMock:
        def configure(self, **kwargs):
            pass
            
        class GenerativeModel:
            def __init__(self, *args, **kwargs):
                pass
                
            def generate_content(self, *args, **kwargs):
                return type('', (), {
                    'text': 'This is a mock response from the AI agent for development/testing purposes.'
                })()
    
    genai = GenAIMock()
    
    # Mock exception class
    class GenAIException(Exception):
        pass
    
    # Mock HarmCategory and HarmBlockThreshold classes
    class HarmCategory:
        HARM_CATEGORY_HARASSMENT = "HARM_CATEGORY_HARASSMENT"
        HARM_CATEGORY_HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
        HARM_CATEGORY_SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
        HARM_CATEGORY_DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"
        
    class HarmBlockThreshold:
        BLOCK_NONE = "BLOCK_NONE"
        BLOCK_LOW = "BLOCK_LOW"
        BLOCK_MEDIUM = "BLOCK_MEDIUM"
        BLOCK_HIGH = "BLOCK_HIGH"
    
    print("WARNING: Using mock Google Generative AI implementation for development/testing purposes.")

from ..config import settings
from ..models.document import OCRResult
from ..models.trf import PatientReport
from ..schemas.trf_schema import get_field_value, set_field_value, REQUIRED_FIELDS
from .knowledge_base import KNOWLEDGE_BASE


class AgentReasoning:
    """Enhanced reasoning engine for the AI agent with patient context awareness."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the agent reasoning engine."""
        self.api_key = api_key or settings.GEMINI_API_KEY
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        # Use Gemini Pro model for text processing
        self.model = genai.GenerativeModel('gemini-pro')
        self.knowledge_base = KNOWLEDGE_BASE
        
        # Configure safety settings (optional)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    async def analyze_ocr_result(self, ocr_result: OCRResult, trf_data: Dict[str, Any], 
                               existing_patient_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze OCR results and TRF data to identify issues and make suggestions.
        Now with patient context awareness.
        
        Args:
            ocr_result: OCR result with extracted text
            trf_data: Current TRF data
            existing_patient_data: Optional existing patient data for context
            
        Returns:
            Analysis results with suggestions
        """
        # Identify missing required fields
        missing_fields = []
        for field in REQUIRED_FIELDS:
            value = get_field_value(trf_data, field)
            if value in (None, "", []):
                # Check if available in existing patient data
                if existing_patient_data:
                    existing_value = get_field_value(existing_patient_data, field)
                    if existing_value not in (None, "", []):
                        # Not missing if we have it in existing data
                        continue
                missing_fields.append(field)
        
        # Identify low-confidence fields
        low_confidence_fields = []
        for field_path, confidence in trf_data.get("extracted_fields", {}).items():
            if confidence < 0.7:
                low_confidence_fields.append(field_path)
        
        name_conflict = None
        hospital_conflict = None
        if existing_patient_data:
            name_conflict = check_name_conflict(trf_data, existing_patient_data, self.knowledge_base["templates"])
            hospital_conflict = check_hospital_conflict(trf_data, existing_patient_data, self.knowledge_base["templates"])
        
        # Prepare the analysis results
        analysis_results = {
            "missing_fields": missing_fields,
            "low_confidence_fields": low_confidence_fields,
            "suggestions": [],
            "completion_percentage": self._calculate_completion_percentage(trf_data, existing_patient_data),
            "analysis_time": time.time(),
            "name_conflict": name_conflict,
            "hospital_conflict": hospital_conflict
        }
        
        # Generate suggestions for missing fields using Gemini AI
        if missing_fields:
            suggestions = await self._generate_field_suggestions(ocr_result.text, missing_fields, 
                                                            trf_data, existing_patient_data)
            analysis_results["suggestions"] = suggestions
        
        return analysis_results
    
    async def query_agent(self, query: str, ocr_text: str, trf_data: Dict[str, Any], 
                        existing_patient_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Query the agent with a natural language question.
        Now with patient context awareness.
        
        Args:
            query: User query
            ocr_text: OCR extracted text
            trf_data: Current TRF data
            existing_patient_data: Optional existing patient data for context
            
        Returns:
            Agent response
        """
        try:
            # Prepare context for the agent
            context = self._prepare_agent_context(ocr_text, trf_data, existing_patient_data)
            
            # Create the full prompt with query, context, and knowledge base
            prompt = self.create_agent_prompt(query, context)
            
            # Call Gemini model
            try:
                # Add system message as part of the prompt
                system_message = "You are an expert medical form assistant that helps extract and validate information from medical documents. You carefully analyze form contents and help users complete Test Requisition Forms (TRFs) accurately."
                full_prompt = f"{system_message}\n\n{prompt}"
                
                response = self.model.generate_content(
                    full_prompt,
                    safety_settings=self.safety_settings
                )
                
                # Extract the response text
                response_text = response.text
            except (AttributeError, IndexError, Exception) as e:
                # Fallback in case API structure changes
                response_text = f"I apologize, but I couldn't analyze the document properly. The OCR service might need to be updated. Error: {str(e)}"
            
            # Parse any suggested actions from the response
            suggested_actions = self._extract_suggested_actions(response_text)
            
            return {
                "query": query,
                "response": response_text,
                "suggested_actions": suggested_actions,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "query": query,
                "response": f"Error querying the agent: {str(e)}",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def suggest_field_value(self, field_path: str, ocr_text: str, trf_data: Dict[str, Any],
                              existing_patient_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Suggest a value for a specific field based on OCR text.
        Now with patient context awareness and proper array handling.
        """
        try:
            # Get field description
            field_description = self.knowledge_base["field_descriptions"].get(field_path, f"Field {field_path}")

            # Check if field value exists in existing patient data
            existing_value = None
            if existing_patient_data:
                # Handle array-like field paths (e.g., Sample.0.sampleType or Sample[0].sampleType)
                if '[' in field_path or ('.' in field_path and field_path.split('.')[-1].isdigit()):
                    parts = field_path.replace('[', '.').replace(']', '').split('.')
                    existing_value = existing_patient_data
                    try:
                        for part in parts:
                            if part.isdigit():
                                existing_value = existing_value[int(part)] if isinstance(existing_value, list) else None
                            else:
                                existing_value = existing_value.get(part) if isinstance(existing_value, dict) else None
                            if existing_value is None:
                                break
                    except (IndexError, TypeError, KeyError):
                        existing_value = None
                else:
                    existing_value = get_field_value(existing_patient_data, field_path)

                if existing_value not in (None, "", []):
                    return {
                        "field_path": field_path,
                        "suggested_value": existing_value,
                        "confidence": 0.95,
                        "reasoning": f"Value retrieved from existing patient data",
                        "timestamp": time.time()
                    }

            # Create context-aware prompt
            context_info = ""
            if existing_patient_data:
                context_info = "You also have access to some existing patient information which may be helpful context."
                related_fields = self._get_related_fields(field_path, existing_patient_data)
                if related_fields:
                    context_info += "\n\nRelated patient information that might help:\n"
                    for rel_field, rel_value in related_fields.items():
                        context_info += f"- {rel_field}: {rel_value}\n"

            system_message = "You are an expert at extracting specific information from medical documents with awareness of patient context."
            prompt = f"""
    I need to extract the value for the field '{field_path}' ({field_description}) from the following OCR text:

    {ocr_text[:5000]}  # Limit text length

    {context_info}

    Based on the text above, what is the most likely value for '{field_path}'?
    Please only respond with the extracted value or 'Not found' if you can't find it in the text.
    Also state your confidence level (0-100%) for the extracted value.
    Format your response as:
    VALUE: [extracted value]
    CONFIDENCE: [0-100]
    REASONING: [brief explanation]
    """

            # Call Gemini AI
            try:
                response = self.model.generate_content(
                    f"{system_message}\n\n{prompt}",
                    safety_settings=self.safety_settings
                )
                response_text = response.text
            except (AttributeError, IndexError, Exception) as e:
                response_text = f"VALUE: Not found\nCONFIDENCE: 0\nREASONING: Unable to process the OCR text. Error: {str(e)}"

            # Parse the response
            value_match = re.search(r"VALUE:\s*(.*?)(?:\n|$)", response_text)
            confidence_match = re.search(r"CONFIDENCE:\s*(\d+)(?:\n|$)", response_text)
            reasoning_match = re.search(r"REASONING:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)

            extracted_value = value_match.group(1).strip() if value_match else None
            confidence = int(confidence_match.group(1)) / 100 if confidence_match else 0.5
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."

            if extracted_value and extracted_value.lower() in ("not found", "none", "n/a", "unknown"):
                extracted_value = None
                confidence = 0.0

            return {
                "field_path": field_path,
                "suggested_value": extracted_value,
                "confidence": confidence,
                "reasoning": reasoning,
                "timestamp": time.time()
            }

        except Exception as e:
            return {
                "field_path": field_path,
                "error": str(e),
                "timestamp": time.time()
            }

    
    async def _generate_field_suggestions(self, ocr_text: str, missing_fields: List[str], 
                                       trf_data: Dict[str, Any], 
                                       existing_patient_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Generate suggestions for missing fields.
        Now with patient context awareness.
        
        Args:
            ocr_text: OCR extracted text
            missing_fields: List of missing field paths
            trf_data: Current TRF data
            existing_patient_data: Optional existing patient data for context
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # For each missing required field, generate a suggestion
        for field_path in missing_fields[:5]:  # Limit to 5 fields for performance
            suggestion = await self.suggest_field_value(field_path, ocr_text, trf_data, existing_patient_data)
            if "error" not in suggestion and suggestion["suggested_value"]:
                suggestions.append(suggestion)
        
        return suggestions
    
    def _calculate_completion_percentage(self, trf_data: Dict[str, Any], 
                                       existing_patient_data: Dict[str, Any] = None) -> float:
        """
        Calculate the percentage of required fields that have been completed.
        Now with patient context awareness.
        """
        if not REQUIRED_FIELDS:
            return 1.0
            
        completed_count = 0
        for field in REQUIRED_FIELDS:
            # Check in TRF data first
            value = get_field_value(trf_data, field)
            if value not in (None, "", []):
                completed_count += 1
                continue
                
            # If not in TRF data, check in existing patient data
            if existing_patient_data:
                existing_value = get_field_value(existing_patient_data, field)
                if existing_value not in (None, "", []):
                    completed_count += 1
                
        return completed_count / len(REQUIRED_FIELDS)
    
    def _prepare_agent_context(self, ocr_text: str, trf_data: Dict[str, Any],
                             existing_patient_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prepare context information for the agent.
        Now with patient context awareness.
        """
        # Calculate completion statistics
        completion_percentage = self._calculate_completion_percentage(trf_data, existing_patient_data)
        
        # Identify missing required fields
        missing_fields = []
        for field in REQUIRED_FIELDS:
            # Check in both TRF data and existing patient data
            trf_value = get_field_value(trf_data, field)
            existing_value = None
            if existing_patient_data:
                existing_value = get_field_value(existing_patient_data, field)
                
            if trf_value in (None, "", []) and (existing_value in (None, "", []) or existing_value is None):
                missing_fields.append(field)
        
        # Create context object
        context = {
            "ocr_text_sample": ocr_text[:2000] + ("..." if len(ocr_text) > 2000 else ""),
            "trf_data": trf_data,
            "existing_patient_data": existing_patient_data,
            "completion_percentage": completion_percentage,
            "missing_fields": missing_fields,
            "extracted_fields": trf_data.get("extracted_fields", {}),
            "timestamp": time.time()
        }
        
        return context
    
    def create_agent_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """
        Create a prompt for the agent based on the query and context.
        Now with improved patient context awareness.
        """
        # Format TRF data as a readable summary
        trf_summary = json.dumps(context["trf_data"], indent=2)
        
        # Include existing patient data if available
        existing_data_section = ""
        if context.get("existing_patient_data"):
            existing_data_summary = json.dumps(context["existing_patient_data"], indent=2)
            existing_data_section = f"""
Existing patient data:
{existing_data_summary}

When analyzing the OCR text, please consider this existing patient data as context.
The OCR extracted data should be used to supplement or correct the existing data,
not replace it entirely. Give priority to existing data when the OCR extraction has
low confidence or is ambiguous.
"""
        
        # Format missing fields as a list
        missing_fields_list = "\n".join([f"- {field}" for field in context["missing_fields"]])
        
        # Construct the prompt
        prompt = f"""
I need help with a Test Requisition Form (TRF). Here's my question:

{query}

Here's the OCR text extract from the document:
{context["ocr_text_sample"]}

Current TRF data from OCR extraction:
{trf_summary}

{existing_data_section}

Missing required fields:
{missing_fields_list}

Completion percentage: {context["completion_percentage"]:.0%}

Please analyze this information and respond to my query. If you can suggest specific field values or corrections, format them like this:
SUGGESTED_ACTION: update_field
FIELD_PATH: [field path]
VALUE: [suggested value]
CONFIDENCE: [0-100]
REASONING: [explanation]
"""
        
        return prompt
    
    def _extract_suggested_actions(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract suggested actions from the agent response."""
        # Keep your existing implementation
        # This doesn't need modification for patient context awareness
        
        suggested_actions = []
        
        # Look for action blocks in the response
        action_blocks = re.finditer(
            r"SUGGESTED_ACTION:\s*(.*?)\n"
            r"FIELD_PATH:\s*(.*?)\n"
            r"VALUE:\s*(.*?)\n"
            r"CONFIDENCE:\s*(\d+)\n"
            r"REASONING:\s*(.*?)(?=SUGGESTED_ACTION:|$)",
            response_text, 
            re.DOTALL
        )
        
        for match in action_blocks:
            action_type = match.group(1).strip()
            field_path = match.group(2).strip()
            value = match.group(3).strip()
            confidence = int(match.group(4).strip()) / 100
            reasoning = match.group(5).strip()
            
            suggested_actions.append({
                "type": action_type,
                "field_path": field_path,
                "value": value,
                "confidence": confidence,
                "reasoning": reasoning
            })
        
        return suggested_actions
        
    def _get_related_fields(self, field_path: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get related fields from patient data that might help with the current field.
        This is used to provide context for field suggestions.
        
        Args:
            field_path: Path to the field being suggested
            patient_data: Existing patient data
            
        Returns:
            Dictionary of related field names and values
        """
        related_fields = {}
        
        # Map of related fields by category
        field_relations = {
            "patientInformation": [
                "patientInformation.patientName.firstName",
                "patientInformation.patientName.middleName",
                "patientInformation.patientName.lastName",
                "patientInformation.gender",
                "patientInformation.dob",
                "patientInformation.age",
                "patientInformation.mrnUhid",
                "patientInformation.patientInformationPhoneNumber",
                "patientInformation.email"
            ],
            "clinicalSummary": [
                "clinicalSummary.primaryDiagnosis",
                "clinicalSummary.initialDiagnosisStage",
                "clinicalSummary.currentDiagnosis",
                "clinicalSummary.diagnosisDate",
                "clinicalSummary.Immunohistochemistry.er",
                "clinicalSummary.Immunohistochemistry.pr",
                "clinicalSummary.Immunohistochemistry.her2neu",
                "clinicalSummary.Immunohistochemistry.ki67"
            ],
            "hospital": [
                "hospital.hospitalName",
                "hospital.hospitalID",
                "hospital.hospitalAddress",
                "hospital.city",
                "hospital.state",
                "hospital.country",
                "hospital.postalCode"
            ],
            "physician": [
                "physician.physicianName",
                "physician.physicianSpecialty",
                "physician.physicianPhoneNumber",
                "physician.physicianEmail"
            ],
            "FamilyHistory": [
                "FamilyHistory.familyHistoryOfAnyCancer"
            ]
        }
        
        # Find which category the field belongs to
        field_category = None
        for category, fields in field_relations.items():
            if field_path.startswith(category):
                field_category = category
                break
                
        if field_category:
            # Add all related fields from the same category
            for related_field in field_relations[field_category]:
                if related_field != field_path:  # Don't include the field itself
                    value = get_field_value(patient_data, related_field)
                    if value not in (None, "", []):
                        # Get field description if available
                        field_desc = self.knowledge_base["field_descriptions"].get(
                            related_field, related_field.split('.')[-1]
                        )
                        related_fields[field_desc] = value
        
        return related_fields


# Create a singleton instance
agent_reasoning = AgentReasoning()
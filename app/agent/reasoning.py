"""Reasoning engine for the AI agent to analyze TRF data and make suggestions."""

import json
import time
import re
from typing import Dict, List, Any, Tuple, Optional
import asyncio

# Import Mistral with error handling
try:
    # Try importing from the latest package structure
    from mistralai.client import MistralClient as Mistral
    from mistralai.exceptions import MistralException
except ImportError:
    try:
        # Try the older package structure
        from mistralai import Mistral
        from mistralai.exceptions import MistralException
    except ImportError:
        # Fallback to a mock implementation for development/testing
        class Mistral:
            def __init__(self, api_key=None):
                self.api_key = api_key
                
            def chat(self, model=None, messages=None):
                return type('', (), {
                    'choices': [
                        type('', (), {
                            'message': type('', (), {
                                'content': 'This is a mock response from the AI agent for development/testing purposes.'
                            })
                        })
                    ]
                })()
        
        class MistralException(Exception):
            pass
        
        print("WARNING: Using mock Mistral AI implementation for development/testing purposes.")

from ..config import settings
from ..models.document import OCRResult
from ..models.trf import PatientReport
from ..schemas.trf_schema import get_field_value, set_field_value, REQUIRED_FIELDS
from .knowledge_base import KNOWLEDGE_BASE


class AgentReasoning:
    """Reasoning engine for the AI agent."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the agent reasoning engine."""
        self.api_key = api_key or settings.MISTRAL_API_KEY
        self.client = Mistral(api_key=self.api_key)
        self.knowledge_base = KNOWLEDGE_BASE
    
    async def analyze_ocr_result(self, ocr_result: OCRResult, trf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze OCR results and TRF data to identify issues and make suggestions.
        
        Args:
            ocr_result: OCR result with extracted text
            trf_data: Current TRF data
            
        Returns:
            Analysis results with suggestions
        """
        # Identify missing required fields
        missing_fields = []
        for field in REQUIRED_FIELDS:
            value = get_field_value(trf_data, field)
            if value in (None, "", []):
                missing_fields.append(field)
        
        # Identify low-confidence fields
        low_confidence_fields = []
        for field_path, confidence in trf_data.get("extracted_fields", {}).items():
            if confidence < 0.7:
                low_confidence_fields.append(field_path)
        
        # Prepare the analysis results
        analysis_results = {
            "missing_fields": missing_fields,
            "low_confidence_fields": low_confidence_fields,
            "suggestions": [],
            "completion_percentage": self._calculate_completion_percentage(trf_data),
            "analysis_time": time.time()
        }
        
        # Generate suggestions for missing fields using Mistral AI
        if missing_fields:
            suggestions = await self._generate_field_suggestions(ocr_result.text, missing_fields, trf_data)
            analysis_results["suggestions"] = suggestions
        
        return analysis_results
    
    async def query_agent(self, query: str, ocr_text: str, trf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query the agent with a natural language question.
        
        Args:
            query: User query
            ocr_text: OCR extracted text
            trf_data: Current TRF data
            
        Returns:
            Agent response
        """
        try:
            # Prepare context for the agent
            context = self._prepare_agent_context(ocr_text, trf_data)
            
            # Create the full prompt with query, context, and knowledge base
            prompt = self._create_agent_prompt(query, context)
            
            # Call Mistral AI model
            try:
                response = self.client.chat(
                    model="mistral-large-latest",
                    messages=[
                        {"role": "system", "content": "You are an expert medical form assistant that helps extract and validate information from medical documents. You carefully analyze form contents and help users complete Test Requisition Forms (TRFs) accurately."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Extract the response text
                response_text = response.choices[0].message.content
            except (AttributeError, IndexError):
                # Fallback in case API structure changes
                response_text = "I apologize, but I couldn't analyze the document properly. The OCR service might need to be updated."
            
            # Parse any suggested actions from the response
            suggested_actions = self._extract_suggested_actions(response_text)
            
            return {
                "query": query,
                "response": response_text,
                "suggested_actions": suggested_actions,
                "timestamp": time.time()
            }
            
        except MistralException as e:
            return {
                "query": query,
                "response": f"Error querying the agent: {str(e)}",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def suggest_field_value(self, field_path: str, ocr_text: str, trf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest a value for a specific field based on OCR text.
        
        Args:
            field_path: Path to the field
            ocr_text: OCR extracted text
            trf_data: Current TRF data
            
        Returns:
            Suggestion result
        """
        try:
            # Get field description
            field_description = self.knowledge_base["field_descriptions"].get(field_path, f"Field {field_path}")
            
            # Create prompt for the specific field
            prompt = f"""
I need to extract the value for the field '{field_path}' ({field_description}) from the following OCR text:

{ocr_text[:5000]}  # Limit text length

Based on the text above, what is the most likely value for '{field_path}'?
Please only respond with the extracted value or 'Not found' if you can't find it in the text.
Also state your confidence level (0-100%) for the extracted value.
Format your response as:
VALUE: [extracted value]
CONFIDENCE: [0-100]
REASONING: [brief explanation]
"""
            
            # Call Mistral AI model
            try:
                response = self.client.chat(
                    model="mistral-large-latest",
                    messages=[
                        {"role": "system", "content": "You are an expert at extracting specific information from medical documents."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Extract the response text
                response_text = response.choices[0].message.content
            except (AttributeError, IndexError):
                # Fallback in case API structure changes
                response_text = "VALUE: Not found\nCONFIDENCE: 0\nREASONING: Unable to process the OCR text."
            
            # Parse the response to extract value and confidence
            value_match = re.search(r"VALUE:\s*(.*?)(?:\n|$)", response_text)
            confidence_match = re.search(r"CONFIDENCE:\s*(\d+)(?:\n|$)", response_text)
            reasoning_match = re.search(r"REASONING:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
            
            extracted_value = value_match.group(1).strip() if value_match else None
            confidence = int(confidence_match.group(1)) / 100 if confidence_match else 0.5
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."
            
            # Normalize "Not found" responses
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
            
        except MistralException as e:
            return {
                "field_path": field_path,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _generate_field_suggestions(self, ocr_text: str, missing_fields: List[str], trf_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate suggestions for missing fields.
        
        Args:
            ocr_text: OCR extracted text
            missing_fields: List of missing field paths
            trf_data: Current TRF data
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # For each missing required field, generate a suggestion
        for field_path in missing_fields[:5]:  # Limit to 5 fields for performance
            suggestion = await self.suggest_field_value(field_path, ocr_text, trf_data)
            if "error" not in suggestion and suggestion["suggested_value"]:
                suggestions.append(suggestion)
        
        return suggestions
    
    def _calculate_completion_percentage(self, trf_data: Dict[str, Any]) -> float:
        """Calculate the percentage of required fields that have been completed."""
        if not REQUIRED_FIELDS:
            return 1.0
            
        completed_count = 0
        for field in REQUIRED_FIELDS:
            value = get_field_value(trf_data, field)
            if value not in (None, "", []):
                completed_count += 1
                
        return completed_count / len(REQUIRED_FIELDS)
    
    def _prepare_agent_context(self, ocr_text: str, trf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context information for the agent."""
        # Calculate completion statistics
        completion_percentage = self._calculate_completion_percentage(trf_data)
        
        # Identify missing required fields
        missing_fields = []
        for field in REQUIRED_FIELDS:
            value = get_field_value(trf_data, field)
            if value in (None, "", []):
                missing_fields.append(field)
        
        # Create context object
        context = {
            "ocr_text_sample": ocr_text[:2000] + ("..." if len(ocr_text) > 2000 else ""),
            "trf_data": trf_data,
            "completion_percentage": completion_percentage,
            "missing_fields": missing_fields,
            "extracted_fields": trf_data.get("extracted_fields", {}),
            "timestamp": time.time()
        }
        
        return context
    
    def _create_agent_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Create a prompt for the agent based on the query and context."""
        # Format TRF data as a readable summary
        trf_summary = json.dumps(context["trf_data"], indent=2)
        
        # Format missing fields as a list
        missing_fields_list = "\n".join([f"- {field}" for field in context["missing_fields"]])
        
        # Construct the prompt
        prompt = f"""
I need help with a Test Requisition Form (TRF). Here's my question:

{query}

Here's the OCR text extract from the document:
{context["ocr_text_sample"]}

Current TRF data summary:
{trf_summary}

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


# Create a singleton instance
agent_reasoning = AgentReasoning()

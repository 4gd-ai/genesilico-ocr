import time
import os
import asyncio
from typing import Dict, List, Any, Tuple, Optional
import json

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from ..models.document import OCRResult
from ..agent.knowledge_base import FIELD_DESCRIPTIONS, KNOWLEDGE_BASE

class AIFieldExtractor:
    """Extract fields from OCR results using LangChain AI instead of regex patterns."""
    
    def __init__(self, ocr_result: OCRResult, model_name: str = "gpt-4o", temperature: float = 0.0, existing_patient_data: Optional[Dict] = None):
        """Initialize the AI field extractor with OCR results and optional patient context."""
        self.ocr_result = ocr_result
        self.extracted_data = {}
        self.confidence_scores = {}
        self.existing_patient_data = existing_patient_data
        self.extraction_stats = {
            "total_fields": 0,
            "extracted_fields": 0,
            "high_confidence_fields": 0,
            "low_confidence_fields": 0
        }
        
        # Get API key from environment variables or settings
        try:
            from ..config import settings
            api_key = os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY
            if not api_key:
                print("WARNING: OPENAI_API_KEY is not set. Field extraction will fail.")
                raise ValueError("OPENAI_API_KEY is not set in environment or settings")
                
            self.llm = ChatOpenAI(model_name=model_name, temperature=temperature, api_key=api_key)
            print(f"Successfully initialized ChatOpenAI with model: {model_name}")
        except Exception as e:
            print(f"ERROR initializing ChatOpenAI: {str(e)}")
            # Still raise the exception so the document processor can handle it properly
            raise
    
    def _create_extraction_prompt(self, ocr_text: str) -> ChatPromptTemplate:
        """Create a prompt for the AI to extract fields from OCR text."""
        # Create a system message that explains the task and provides schema information
        system_template = """
        You are an AI assistant specialized in extracting structured medical information from OCR text.
        Your task is to extract relevant fields for a Test Requisition Form (TRF) from the provided text.
        
        Below is information about the TRF schema and the fields you need to extract:
        
        {schema_overview}
        
        Please extract these fields from the OCR text:
        {field_descriptions}
        
        {patient_context}
        
        Please follow these guidelines:
        1. Extract each field based on the OCR text.
        2. If a field is not found in the OCR text, leave it empty.
        3. For each extracted field, provide a confidence score between 0.0 and 1.0.
        4. Format dates as MM/DD/YYYY when possible.
        5. Normalize gender values to "Male", "Female", or "Other".
        
        Return your response as a JSON object with two main sections:
        1. "extracted_fields": A nested JSON object with the extracted field values
        2. "confidence_scores": A flat dictionary mapping field paths to confidence scores
        
        Example response format:
        ```json
        {{
            "extracted_fields": {{
                "patientID": "12345",
                "patientInformation": {{
                    "patientName": {{
                        "firstName": "John",
                        "middleName": "",
                        "lastName": "Doe"
                    }},
                    "gender": "Male",
                    ...
                }},
                ...
            }},
            "confidence_scores": {{
                "patientID": 0.9,
                "patientInformation.patientName.firstName": 0.95,
                ...
            }}
        }}
        ```
        """
        
        system_message = SystemMessagePromptTemplate.from_template(system_template)
        
        # Create a human message with the OCR text
        human_template = """
        Here is the OCR text extracted from a medical document:
        
        ```
        {ocr_text}
        ```
        
        Please extract the TRF fields from this text according to the guidelines provided.
        """
        
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        
        # Combine messages into a ChatPromptTemplate
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        return chat_prompt
    
    def _add_field_to_context(self, context_str, data, field_path, label=None):
        """
        Safely add a field to the context string, handling arrays properly.
        
        Args:
            context_str: The context string to append to
            data: The data dictionary to extract from
            field_path: The path to the field in dot notation
            label: The label to use for the field (defaults to the last part of the path)
            
        Returns:
            Updated context string
        """
        # Import get_field_value for safe field access that handles arrays properly
        from ..schemas.trf_schema import get_field_value
        
        value = get_field_value(data, field_path)
        if value not in (None, "", []):
            field_label = label or field_path.split('.')[-1]
            context_str += f"{field_label}: {value}\n"
        
        return context_str
    
    async def extract_fields(self) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
        """
        Extract fields from OCR text using LLM.
        
        Returns:
            Tuple of (extracted_data, confidence_scores, extraction_stats)
        """
        start_time = time.time()
        
        # Initialize a new PatientReport with default values
        trf_data = {"patientID": f"TEMP-{int(time.time())}"}
        
        # Get the full OCR text
        ocr_text = self.ocr_result.text
        
        # Create the prompt
        prompt = self._create_extraction_prompt(ocr_text)
        
        # Create a chain with the LLM
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Prepare patient context if available
        patient_context = ""
        if self.existing_patient_data:
            patient_context = """
            EXISTING PATIENT CONTEXT:
            You have access to some existing patient data that may help with your extraction.
            Use this information as context to help fill gaps in the OCR text or confirm extracted values.
            
            """
            
            # Use the safe helper function for key fields
            for field_path, label in [
                ("patientInformation.patientName.firstName", "First Name"),
                ("patientInformation.patientName.lastName", "Last Name"),
                ("patientInformation.gender", "Gender"),
                ("patientInformation.dob", "Date of Birth"),
                ("patientInformation.age", "Age"),
                ("patientInformation.patientInformationPhoneNumber", "Phone"),
                ("clinicalSummary.primaryDiagnosis", "Primary Diagnosis"),
                ("hospital.hospitalName", "Hospital"),
                ("physician.physicianName", "Physician")
            ]:
                patient_context = self._add_field_to_context(
                    patient_context, 
                    self.existing_patient_data, 
                    field_path, 
                    label
                )
            
            print(f"Added patient context to extraction prompt:\n{patient_context}")
        
        # Run the chain with the OCR text, schema information, and patient context
        try:
            print(f"Running LLM chain to extract fields from OCR text of length: {len(ocr_text)}")
            response = await chain.arun(
                ocr_text=ocr_text,
                schema_overview=KNOWLEDGE_BASE["schema_overview"],
                field_descriptions=json.dumps(FIELD_DESCRIPTIONS, indent=2),
                patient_context=patient_context
            )
            print(f"LLM chain completed successfully")
        except Exception as e:
            print(f"!!! ERROR IN LLM CHAIN EXECUTION !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            # Re-raise the exception so it can be caught by the outer try-except
            raise
        
        # Parse the JSON response
        try:
            # Print original response for debugging
            print(f"\n=== Raw LLM Response ===")
            print(f"Response length: {len(response)}")
            print(f"Response preview: {response[:500]}...\n")
            
            # Find the JSON part in the response (it might be wrapped in markdown code blocks)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_response = response[json_start:json_end]
                print(f"Found JSON at positions {json_start} to {json_end}")
                extraction_result = json.loads(json_response)
            else:
                print(f"No JSON markers found, trying to parse entire response as JSON")
                extraction_result = json.loads(response)
                
            # Extract the fields and confidence scores
            self.extracted_data = extraction_result.get("extracted_fields", {})
            self.confidence_scores = extraction_result.get("confidence_scores", {})
            
            print(f"\n=== Extraction Data Parsed ===")
            print(f"Extracted fields: {list(self.extracted_data.keys()) if self.extracted_data else 'None'}")
            print(f"Confidence scores: {list(self.confidence_scores.keys()) if self.confidence_scores else 'None'}")
            
            # Merge the extracted data into the TRF data
            self._merge_extracted_data(trf_data, self.extracted_data)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response: {response}")
            
        # Update extraction statistics
        self.extraction_stats["total_fields"] = len(FIELD_DESCRIPTIONS)
        self.extraction_stats["extracted_fields"] = len(self.confidence_scores)
        self.extraction_stats["high_confidence_fields"] = sum(1 for conf in self.confidence_scores.values() if conf >= 0.7)
        self.extraction_stats["low_confidence_fields"] = sum(1 for conf in self.confidence_scores.values() if conf < 0.7)
        self.extraction_stats["extraction_time"] = time.time() - start_time
        
        return trf_data, self.confidence_scores, self.extraction_stats
    
    def _merge_extracted_data(self, target: Dict[str, Any], source: Dict[str, Any], prefix: str = ""):
        """
        Recursively merge extracted data into the target dictionary.
        
        Args:
            target: The target dictionary to merge into
            source: The source dictionary to merge from
            prefix: The current field path prefix
        """
        # Import the safe field setter
        from ..schemas.trf_schema import set_field_value
        
        for key, value in source.items():
            current_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # If the key doesn't exist in the target or isn't a dict, initialize it
                if key not in target or not isinstance(target[key], dict):
                    target[key] = {}
                
                # Recursively merge the nested dictionary
                self._merge_extracted_data(target[key], value, current_path)
            elif isinstance(value, list):
                # Handle arrays (like Sample) safely
                if key not in target:
                    target[key] = []
                
                # Make sure target[key] is actually a list
                if not isinstance(target[key], list):
                    target[key] = []
                
                # Ensure the target array has enough elements
                while len(target[key]) < len(value):
                    target[key].append({} if value and isinstance(value[0], dict) else None)
                
                # Merge each item in the array
                for i, item in enumerate(value):
                    try:
                        if isinstance(item, dict):
                            # If the target array item isn't a dict, make it one
                            if not isinstance(target[key][i], dict):
                                target[key][i] = {}
                                
                            item_path = f"{current_path}.{i}"
                            self._merge_extracted_data(target[key][i], item, item_path)
                        else:
                            target[key][i] = item
                    except (IndexError, TypeError) as e:
                        print(f"Error merging array item {i} at path {current_path}: {str(e)}")
                        # Fix the array and try again
                        if i >= len(target[key]):
                            target[key].append({} if isinstance(item, dict) else None)
                        if isinstance(item, dict) and not isinstance(target[key][i], dict):
                            target[key][i] = {}
                        
                        # Try again
                        if isinstance(item, dict):
                            item_path = f"{current_path}.{i}"
                            self._merge_extracted_data(target[key][i], item, item_path)
                        else:
                            target[key][i] = item
            else:
                # Set the value directly for non-dict, non-list values
                target[key] = value
    
    async def extract_with_focused_agents(self) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
        """
        Extract fields using multiple focused agents for different sections.
        
        Returns:
            Tuple of (extracted_data, confidence_scores, extraction_stats)
        """
        # Initialize timing
        start_time = time.time()
        
        # Initialize a new PatientReport with default values
        trf_data = {"patientID": f"TEMP-{int(time.time())}"}
        
        # Extract each section in parallel
        tasks = [
            self._extract_patient_info(),
            self._extract_clinical_summary(),
            self._extract_physician_info(),
            self._extract_sample_info(),
            self._extract_hospital_info()
        ]
        
        # Run all extraction tasks and merge results
        section_results = await asyncio.gather(*tasks)
        
        # Merge all section results
        for section_data, section_confidence in section_results:
            self._merge_extracted_data(trf_data, section_data)
            self.confidence_scores.update(section_confidence)
        
        # Update extraction statistics
        self.extraction_stats["total_fields"] = len(FIELD_DESCRIPTIONS)
        self.extraction_stats["extracted_fields"] = len(self.confidence_scores)
        self.extraction_stats["high_confidence_fields"] = sum(1 for conf in self.confidence_scores.values() if conf >= 0.7)
        self.extraction_stats["low_confidence_fields"] = sum(1 for conf in self.confidence_scores.values() if conf < 0.7)
        self.extraction_stats["extraction_time"] = time.time() - start_time
        
        return trf_data, self.confidence_scores, self.extraction_stats
    
    async def _extract_section(self, section_name: str, fields_to_extract: List[str]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Extract a specific section using a focused agent.
        
        Args:
            section_name: Name of the section to extract
            fields_to_extract: List of field paths to extract
            
        Returns:
            Tuple of (section_data, section_confidence_scores)
        """
        # Get the full OCR text
        ocr_text = self.ocr_result.text
        
        # Create a system message for this specific section
        system_template = f"""
        You are an AI assistant specialized in extracting {section_name} information from medical documents.
        Your task is to extract only the following fields from the OCR text:
        
        {json.dumps({field: FIELD_DESCRIPTIONS.get(field, "No description available") for field in fields_to_extract}, indent=2)}
        
        For each field, provide:
        1. The extracted value
        2. A confidence score between 0.0 and 1.0
        
        Return your response as a JSON object with two keys:
        1. "extracted_fields": The nested structure with extracted values
        2. "confidence_scores": A flat dictionary with field paths and confidence scores
        """
        
        human_template = f"""
        Here is the OCR text:
        
        ```
        {ocr_text}
        ```
        
        Please extract only the {section_name} fields listed above.
        """
        
        # Create a ChatPromptTemplate
        system_message = SystemMessagePromptTemplate.from_template(system_template)
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        # Create a chain with the LLM
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        
        # Run the chain
        try:
            print(f"Running LLM chain for {section_name} section")
            response = await chain.arun()
            print(f"LLM chain for {section_name} completed successfully")
        except Exception as e:
            print(f"!!! ERROR IN {section_name} LLM CHAIN EXECUTION !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise
        
        # Parse the JSON response
        try:
            # Find the JSON part in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_response = response[json_start:json_end]
                result = json.loads(json_response)
            else:
                result = json.loads(response)
                
            extracted_fields = result.get("extracted_fields", {})
            confidence_scores = result.get("confidence_scores", {})
            
            return extracted_fields, confidence_scores
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response for {section_name}: {e}")
            print(f"Response: {response}")
            return {}, {}
    
    async def _extract_patient_info(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Extract patient information fields."""
        fields = [
            "patientInformation.patientName.firstName",
            "patientInformation.patientName.middleName",
            "patientInformation.patientName.lastName",
            "patientInformation.gender",
            "patientInformation.dob",
            "patientInformation.age",
            "patientInformation.email",
            "patientInformation.patientInformationPhoneNumber",
            "patientInformation.patientInformationAddress",
        ]
        return await self._extract_section("Patient Information", fields)
    
    async def _extract_clinical_summary(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Extract clinical summary fields."""
        fields = [
            "clinicalSummary.primaryDiagnosis",
            "clinicalSummary.initialDiagnosisStage",
            "clinicalSummary.currentDiagnosis",
            "clinicalSummary.diagnosisDate",
            "clinicalSummary.Immunohistochemistry.er",
            "clinicalSummary.Immunohistochemistry.pr",
            "clinicalSummary.Immunohistochemistry.her2neu",
            "clinicalSummary.Immunohistochemistry.ki67",
        ]
        return await self._extract_section("Clinical Summary", fields)
    
    async def _extract_physician_info(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Extract physician information fields."""
        fields = [
            "physician.physicianName",
            "physician.physicianSpecialty",
            "physician.physicianPhoneNumber",
            "physician.physicianEmail",
        ]
        return await self._extract_section("Physician Information", fields)
    
    async def _extract_sample_info(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Extract sample information fields."""
        fields = [
            "Sample.0.sampleType",
            "Sample.0.sampleID",
            "Sample.0.sampleCollectionDate",
            "Sample.0.selectTheTemperatureAtWhichItIsStored",
            "Sample.0.sampleCollectionSite",
        ]
        return await self._extract_section("Sample Information", fields)
    
    async def _extract_hospital_info(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Extract hospital information fields."""
        fields = [
            "hospital.hospitalName",
            "hospital.hospitalAddress",
            "hospital.contactPersonNameHospital",
        ]
        return await self._extract_section("Hospital Information", fields)
    
    def get_field_confidence(self, field_path: str) -> float:
        """Get the confidence score for a specific field."""
        return self.confidence_scores.get(field_path, 0.0)
    
    def get_low_confidence_fields(self, threshold: float = 0.7) -> List[str]:
        """Get a list of fields with confidence below the threshold."""
        return [field for field, conf in self.confidence_scores.items() if conf < threshold]
    
    def get_high_confidence_fields(self, threshold: float = 0.7) -> List[str]:
        """Get a list of fields with confidence above or equal to the threshold."""
        return [field for field, conf in self.confidence_scores.items() if conf >= threshold]
    
    def convert_sample_to_list(data):
        """
        Convert Sample field from object with numeric keys to a list format
        that Pydantic can validate.
        """
        if 'Sample' in data and isinstance(data['Sample'], dict):
            # Check if it's in the format {"0": {...}, "1": {...}}
            if all(key.isdigit() for key in data['Sample'].keys()):
                # Convert to list based on numeric keys
                sample_list = []
                for key in sorted(data['Sample'].keys(), key=lambda x: int(x)):
                    sample_list.append(data['Sample'][key])
                data['Sample'] = sample_list
                
        return data

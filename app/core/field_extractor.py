import time
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
    
    def __init__(self, ocr_result: OCRResult, model_name: str = "gpt-4o", temperature: float = 0.0):
        """Initialize the AI field extractor with OCR results."""
        self.ocr_result = ocr_result
        self.extracted_data = {}
        self.confidence_scores = {}
        self.extraction_stats = {
            "total_fields": 0,
            "extracted_fields": 0,
            "high_confidence_fields": 0,
            "low_confidence_fields": 0
        }
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature, api_key="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")
    
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
        
        # Run the chain with the OCR text and schema information
        response = await chain.arun(
            ocr_text=ocr_text,
            schema_overview=KNOWLEDGE_BASE["schema_overview"],
            field_descriptions=json.dumps(FIELD_DESCRIPTIONS, indent=2)
        )
        
        # Parse the JSON response
        try:
            # Find the JSON part in the response (it might be wrapped in markdown code blocks)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_response = response[json_start:json_end]
                extraction_result = json.loads(json_response)
            else:
                extraction_result = json.loads(response)
                
            # Extract the fields and confidence scores
            self.extracted_data = extraction_result.get("extracted_fields", {})
            self.confidence_scores = extraction_result.get("confidence_scores", {})
            
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
        for key, value in source.items():
            current_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # If the key doesn't exist in the target or isn't a dict, initialize it
                if key not in target or not isinstance(target[key], dict):
                    target[key] = {}
                
                # Recursively merge the nested dictionary
                self._merge_extracted_data(target[key], value, current_path)
            elif isinstance(value, list):
                # Handle arrays (like Sample)
                if key not in target:
                    target[key] = []
                
                # Ensure the target array has enough elements
                while len(target[key]) < len(value):
                    target[key].append({})
                
                # Merge each item in the array
                for i, item in enumerate(value):
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
        response = await chain.arun()
        
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

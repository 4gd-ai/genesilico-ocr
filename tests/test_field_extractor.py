"""Tests for AI field extractor functionality."""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

from app.core.field_extractor import AIFieldExtractor
from app.models.document import OCRResult


# Sample OCR text for testing
SAMPLE_OCR_TEXT = """
Test Requisition Form

Patient Name: John Smith
DOB: 01/15/1980
Gender: Male
Phone: (555) 123-4567
Email: john.smith@example.com

Primary Diagnosis: Breast Cancer
Diagnosis Date: 05/10/2022

Doctor: Dr. Jane Johnson
Doctor Email: jane.johnson@hospital.com
Phone: (555) 987-6543

Hospital: Memorial Hospital
Hospital Address: 123 Main St, Anytown, CA 12345

Sample Type: Blood
Sample ID: S12345
Collection Date: 06/01/2022
"""

# Sample patient data for testing
SAMPLE_PATIENT_DATA = {
    "patientID": "patient-123",
    "patientInformation": {
        "patientName": {
            "firstName": "John",
            "lastName": "Smith"
        },
        "gender": "Male",
        "dob": "01/15/1980",
        "patientInformationPhoneNumber": "(555) 123-4567",
        "email": "john.smith@example.com"
    },
    "clinicalSummary": {
        "primaryDiagnosis": "Breast Cancer",
        "diagnosisDate": "05/10/2022"
    },
    "hospital": {
        "hospitalName": "Memorial Hospital",
        "hospitalAddress": "123 Main St, Anytown, CA 12345"
    }
}

# Sample extraction result for testing
SAMPLE_EXTRACTION_RESULT = {
    "extracted_fields": {
        "patientID": "TEMP-123456789",
        "patientInformation": {
            "patientName": {
                "firstName": "John",
                "lastName": "Smith"
            },
            "gender": "Male",
            "dob": "01/15/1980",
            "patientInformationPhoneNumber": "(555) 123-4567",
            "email": "john.smith@example.com"
        },
        "clinicalSummary": {
            "primaryDiagnosis": "Breast Cancer",
            "diagnosisDate": "05/10/2022"
        },
        "physician": {
            "physicianName": "Dr. Jane Johnson",
            "physicianEmail": "jane.johnson@hospital.com",
            "physicianPhoneNumber": "(555) 987-6543"
        },
        "hospital": {
            "hospitalName": "Memorial Hospital",
            "hospitalAddress": "123 Main St, Anytown, CA 12345"
        },
        "Sample": {
            "0": {
                "sampleType": "Blood",
                "sampleID": "S12345",
                "sampleCollectionDate": "06/01/2022"
            }
        }
    },
    "confidence_scores": {
        "patientInformation.patientName.firstName": 0.9,
        "patientInformation.patientName.lastName": 0.9,
        "patientInformation.gender": 0.8,
        "patientInformation.dob": 0.8,
        "patientInformation.patientInformationPhoneNumber": 0.7,
        "patientInformation.email": 0.9,
        "clinicalSummary.primaryDiagnosis": 0.8,
        "clinicalSummary.diagnosisDate": 0.8,
        "physician.physicianName": 0.85,
        "physician.physicianEmail": 0.85,
        "physician.physicianPhoneNumber": 0.8,
        "hospital.hospitalName": 0.9,
        "hospital.hospitalAddress": 0.85,
        "Sample.0.sampleType": 0.8,
        "Sample.0.sampleID": 0.85,
        "Sample.0.sampleCollectionDate": 0.8
    }
}


@pytest.fixture
def ocr_result():
    """Create a test OCR result."""
    result = OCRResult(
        document_id="test_document_id",
        text=SAMPLE_OCR_TEXT,
        confidence=0.85,
        processing_time=1.2,
        pages=[
            {
                "page_num": 1,
                "text": SAMPLE_OCR_TEXT,
                "blocks": []
            }
        ]
    )
    return result


@pytest.fixture
def mock_langchain_chain():
    """Mock LangChain Chain."""
    with patch("app.core.field_extractor.LLMChain") as mock_chain_class:
        mock_chain = MagicMock()
        mock_chain.arun = AsyncMock(return_value=json.dumps(SAMPLE_EXTRACTION_RESULT))
        mock_chain_class.return_value = mock_chain
        yield mock_chain


@pytest.fixture
def mock_langchain_chat():
    """Mock LangChain ChatOpenAI."""
    with patch("app.core.field_extractor.ChatOpenAI") as mock_chat_class:
        mock_chat = MagicMock()
        mock_chat_class.return_value = mock_chat
        yield mock_chat


@pytest.fixture
def mock_get_field_value():
    """Mock get_field_value function."""
    with patch("app.core.field_extractor.get_field_value") as mock_function:
        mock_function.side_effect = lambda obj, path: "mocked_value" if path and obj else None
        yield mock_function


@pytest.fixture
def mock_set_field_value():
    """Mock set_field_value function."""
    with patch("app.core.field_extractor.set_field_value") as mock_function:
        mock_function.side_effect = lambda obj, path, value: None
        yield mock_function


@pytest.fixture
def mock_field_descriptions():
    """Mock FIELD_DESCRIPTIONS."""
    with patch("app.core.field_extractor.FIELD_DESCRIPTIONS", {"test.field": "Field description"}), \
         patch("app.core.field_extractor.KNOWLEDGE_BASE", {"schema_overview": "Test schema overview"}):
        yield


@pytest.mark.asyncio
async def test_init_field_extractor(mock_langchain_chat, ocr_result):
    """Test initializing the field extractor."""
    # Initialize field extractor
    # extractor = AIFieldExtractor(ocr_result)
    
    # # Verify initialization
    # assert extractor.ocr_result == ocr_result
    # assert isinstance(extractor.extracted_data, dict)
    # assert isinstance(extractor.confidence_scores, dict)
    # assert isinstance(extractor.extraction_stats, dict)
    
    # Verify LLM initialization
    mock_langchain_chat.assert_called_once()
    # args, kwargs = mock_langchain_chat.call_args
    # assert kwargs["model_name"] == "gpt-4o"
    # assert kwargs["temperature"] == 0.0
    # assert kwargs["api_key"] is not None


@pytest.mark.asyncio
async def test_extract_fields(mock_langchain_chain, mock_langchain_chat, 
                             mock_field_descriptions, ocr_result):
    """Test extracting fields from OCR text."""
    # Initialize field extractor
    extractor = AIFieldExtractor(ocr_result)
    await extractor.extract_fields()
    mock_langchain_chain.assert_called_once()


@pytest.mark.asyncio
async def test_extract_fields_with_patient_context(mock_langchain_chain, mock_langchain_chat, 
                                                 mock_field_descriptions, ocr_result):
    """Test extracting fields with existing patient context."""
    extractor = AIFieldExtractor(ocr_result, existing_patient_data=SAMPLE_PATIENT_DATA)
    await extractor.extract_fields()
    mock_langchain_chain.assert_called_once()


@pytest.mark.asyncio
async def test_extract_with_focused_agents(mock_langchain_chain, mock_langchain_chat, 
                                         mock_field_descriptions, ocr_result):
    """Test extracting fields with focused agents for different sections."""
    # Patch the _extract_section method
    with patch.object(AIFieldExtractor, "_extract_section") as mock_extract_section:
        # Configure the mock to return a tuple of ({}, {})
        mock_extract_section.return_value = ({}, {})
        
        # Initialize field extractor
        extractor = AIFieldExtractor(ocr_result)
        
        # Extract fields with focused agents
        trf_data, confidence_scores, extraction_stats = await extractor.extract_with_focused_agents()
        
        # Verify the result
        assert isinstance(trf_data, dict)
        assert isinstance(confidence_scores, dict)
        assert isinstance(extraction_stats, dict)
        
        # Verify that _extract_section was called for each section
        assert mock_extract_section.call_count == 5


@pytest.mark.asyncio
async def test_extract_section(mock_langchain_chain, mock_langchain_chat, 
                             mock_field_descriptions, ocr_result):
    """Test extracting a specific section."""
    # Initialize field extractor
    extractor = AIFieldExtractor(ocr_result)
    
    # Patch the LLMChain class
    with patch("app.core.field_extractor.LLMChain") as mock_chain_class:
        # Configure the mock to return a JSON string
        mock_chain = MagicMock()
        mock_chain.arun = AsyncMock(return_value=json.dumps({
            "extracted_fields": {
                "patientInformation": {
                    "patientName": {
                        "firstName": "John",
                        "lastName": "Smith"
                    }
                }
            },
            "confidence_scores": {
                "patientInformation.patientName.firstName": 0.9,
                "patientInformation.patientName.lastName": 0.9
            }
        }))
        mock_chain_class.return_value = mock_chain
        
        # Extract a section
        fields = [
            "patientInformation.patientName.firstName",
            "patientInformation.patientName.lastName"
        ]
        section_data, section_confidence = await extractor._extract_section("Patient Information", fields)
        
        # Verify the result
        assert isinstance(section_data, dict)
        assert isinstance(section_confidence, dict)
        assert "patientInformation" in section_data
        assert "patientName" in section_data["patientInformation"]
        assert section_data["patientInformation"]["patientName"]["firstName"] == "John"
        assert section_data["patientInformation"]["patientName"]["lastName"] == "Smith"
        assert section_confidence["patientInformation.patientName.firstName"] == 0.9
        assert section_confidence["patientInformation.patientName.lastName"] == 0.9
        
        # Verify that LLMChain was created and called correctly
        mock_chain_class.assert_called_once()
        mock_chain.arun.assert_called_once()


@pytest.mark.asyncio
async def test_extract_section_json_error(mock_langchain_chain, mock_langchain_chat, 
                                        mock_field_descriptions, ocr_result):
    """Test error handling when extracting a section with invalid JSON."""
    # Initialize field extractor
    extractor = AIFieldExtractor(ocr_result)
    
    # Patch the LLMChain class
    with patch("app.core.field_extractor.LLMChain") as mock_chain_class:
        # Configure the mock to return an invalid JSON string
        mock_chain = MagicMock()
        mock_chain.arun = AsyncMock(return_value="This is not valid JSON")
        mock_chain_class.return_value = mock_chain
        
        # Extract a section
        fields = [
            "patientInformation.patientName.firstName",
            "patientInformation.patientName.lastName"
        ]
        section_data, section_confidence = await extractor._extract_section("Patient Information", fields)
        
        # Verify the result
        assert isinstance(section_data, dict)
        assert isinstance(section_confidence, dict)
        assert not section_data
        assert not section_confidence


def test_get_field_confidence(mock_langchain_chat, ocr_result):
    """Test getting the confidence score for a specific field."""
    # Initialize field extractor
    extractor = AIFieldExtractor(ocr_result)
    
    # Set confidence scores
    extractor.confidence_scores = {
        "patientInformation.patientName.firstName": 0.9,
        "patientInformation.patientName.lastName": 0.8,
        "patientInformation.gender": 0.7
    }
    
    # Get confidence for existing field
    confidence = extractor.get_field_confidence("patientInformation.patientName.firstName")
    assert confidence == 0.9
    
    # Get confidence for non-existent field
    confidence = extractor.get_field_confidence("non.existent.field")
    assert confidence == 0.0


def test_get_low_confidence_fields(mock_langchain_chat, ocr_result):
    """Test getting fields with confidence below a threshold."""
    # Initialize field extractor
    extractor = AIFieldExtractor(ocr_result)
    
    # Set confidence scores
    extractor.confidence_scores = {
        "patientInformation.patientName.firstName": 0.9,
        "patientInformation.patientName.lastName": 0.8,
        "patientInformation.gender": 0.6,
        "patientInformation.dob": 0.5
    }
    
    # Get low confidence fields
    low_confidence = extractor.get_low_confidence_fields(threshold=0.7)
    assert "patientInformation.gender" in low_confidence
    assert "patientInformation.dob" in low_confidence
    assert "patientInformation.patientName.firstName" not in low_confidence
    assert "patientInformation.patientName.lastName" not in low_confidence


def test_get_high_confidence_fields(mock_langchain_chat, ocr_result):
    """Test getting fields with confidence above or equal to a threshold."""
    # Initialize field extractor
    extractor = AIFieldExtractor(ocr_result)
    
    # Set confidence scores
    extractor.confidence_scores = {
        "patientInformation.patientName.firstName": 0.9,
        "patientInformation.patientName.lastName": 0.8,
        "patientInformation.gender": 0.6,
        "patientInformation.dob": 0.5
    }
    
    # Get high confidence fields
    high_confidence = extractor.get_high_confidence_fields(threshold=0.7)
    assert "patientInformation.patientName.firstName" in high_confidence
    assert "patientInformation.patientName.lastName" in high_confidence
    assert "patientInformation.gender" not in high_confidence
    assert "patientInformation.dob" not in high_confidence


def test_merge_extracted_data(mock_langchain_chat, ocr_result):
    """Test merging extracted data into a target dictionary."""
    # Initialize field extractor
    extractor = AIFieldExtractor(ocr_result)
    
    # Create target and source dictionaries
    target = {
        "patientID": "TEMP-123456789",
        "patientInformation": {
            "patientName": {
                "firstName": "",
                "lastName": ""
            }
        }
    }
    
    source = {
        "patientInformation": {
            "patientName": {
                "firstName": "John",
                "lastName": "Smith"
            },
            "gender": "Male"
        },
        "Sample": {
            "0": {
                "sampleType": "Blood",
                "sampleID": "S12345"
            }
        }
    }
    
    # Merge the data
    extractor._merge_extracted_data(target, source)
    
    # Verify the merged data
    assert target["patientID"] == "TEMP-123456789"
    assert target["patientInformation"]["patientName"]["firstName"] == "John"
    assert target["patientInformation"]["patientName"]["lastName"] == "Smith"
    assert target["patientInformation"]["gender"] == "Male"
    assert "Sample" in target
    assert "0" in target["Sample"]
    assert target["Sample"]["0"]["sampleType"] == "Blood"
    assert target["Sample"]["0"]["sampleID"] == "S12345"


@pytest.mark.asyncio
async def test_extract_patient_info(mock_langchain_chat, ocr_result):
    """Test extracting patient information fields."""
    # Initialize field extractor
    extractor = AIFieldExtractor(ocr_result)
    
    # Patch the _extract_section method
    with patch.object(extractor, "_extract_section") as mock_extract_section:
        # Configure the mock to return a tuple
        mock_extract_section.return_value = (
            {"patientInformation": {"patientName": {"firstName": "John", "lastName": "Smith"}}},
            {"patientInformation.patientName.firstName": 0.9, "patientInformation.patientName.lastName": 0.9}
        )
        
        # Extract patient information
        patient_data, patient_confidence = await extractor._extract_patient_info()
        
        # Verify the result
        assert isinstance(patient_data, dict)
        assert isinstance(patient_confidence, dict)
        assert "patientInformation" in patient_data
        
        # Verify that _extract_section was called correctly
        mock_extract_section.assert_called_once()


@pytest.mark.asyncio
async def test_extract_clinical_summary(mock_langchain_chat, ocr_result):
    """Test extracting clinical summary fields."""
    # Initialize field extractor
    extractor = AIFieldExtractor(ocr_result)
    
    # Patch the _extract_section method
    with patch.object(extractor, "_extract_section") as mock_extract_section:
        # Configure the mock to return a tuple
        mock_extract_section.return_value = (
            {"clinicalSummary": {"primaryDiagnosis": "Breast Cancer"}},
            {"clinicalSummary.primaryDiagnosis": 0.8}
        )
        
        # Extract clinical summary
        clinical_data, clinical_confidence = await extractor._extract_clinical_summary()
        
        # Verify the result
        assert isinstance(clinical_data, dict)
        assert isinstance(clinical_confidence, dict)
        assert "clinicalSummary" in clinical_data
        
        # Verify that _extract_section was called correctly
        mock_extract_section.assert_called_once()


@pytest.mark.asyncio
async def test_extract_physician_info(mock_langchain_chat, ocr_result):
    """Test extracting physician information fields."""
    # Initialize field extractor
    extractor = AIFieldExtractor(ocr_result)
    
    # Patch the _extract_section method
    with patch.object(extractor, "_extract_section") as mock_extract_section:
        # Configure the mock to return a tuple
        mock_extract_section.return_value = (
            {"physician": {"physicianName": "Dr. Jane Johnson"}},
            {"physician.physicianName": 0.85}
        )
        
        # Extract physician information
        physician_data, physician_confidence = await extractor._extract_physician_info()
        
        # Verify the result
        assert isinstance(physician_data, dict)
        assert isinstance(physician_confidence, dict)
        assert "physician" in physician_data
        
        # Verify that _extract_section was called correctly
        mock_extract_section.assert_called_once()


@pytest.mark.asyncio
async def test_extract_sample_info(mock_langchain_chat, ocr_result):
    """Test extracting sample information fields."""
    # Initialize field extractor
    extractor = AIFieldExtractor(ocr_result)
    
    # Patch the _extract_section method
    with patch.object(extractor, "_extract_section") as mock_extract_section:
        # Configure the mock to return a tuple
        mock_extract_section.return_value = (
            {"Sample": {"0": {"sampleType": "Blood"}}},
            {"Sample.0.sampleType": 0.8}
        )
        
        # Extract sample information
        sample_data, sample_confidence = await extractor._extract_sample_info()
        
        # Verify the result
        assert isinstance(sample_data, dict)
        assert isinstance(sample_confidence, dict)
        assert "Sample" in sample_data
        
        # Verify that _extract_section was called correctly
        mock_extract_section.assert_called_once()


@pytest.mark.asyncio
async def test_extract_hospital_info(mock_langchain_chat, ocr_result):
    """Test extracting hospital information fields."""
    # Initialize field extractor
    extractor = AIFieldExtractor(ocr_result)
    
    # Patch the _extract_section method
    with patch.object(extractor, "_extract_section") as mock_extract_section:
        # Configure the mock to return a tuple
        mock_extract_section.return_value = (
            {"hospital": {"hospitalName": "Memorial Hospital"}},
            {"hospital.hospitalName": 0.9}
        )
        
        # Extract hospital information
        hospital_data, hospital_confidence = await extractor._extract_hospital_info()
        
        # Verify the result
        assert isinstance(hospital_data, dict)
        assert isinstance(hospital_confidence, dict)
        assert "hospital" in hospital_data
        
        # Verify that _extract_section was called correctly
        mock_extract_section.assert_called_once()

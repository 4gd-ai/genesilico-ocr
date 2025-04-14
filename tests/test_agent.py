"""Tests for AI agent functionality."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.agent.reasoning import AgentReasoning
from app.agent.suggestions import AgentSuggestions


# Create test client
client = TestClient(app)


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


# Mock database connections
@pytest.fixture(autouse=True)
def mock_db_connection():
    """Mock MongoDB connection."""
    with patch("app.core.database.connect_to_mongodb") as mock_connect:
        mock_connect.return_value = True
        yield mock_connect


# Mock collections
@pytest.fixture
def mock_collections():
    """Mock database collections."""
    with patch("app.core.database.documents_collection") as mock_docs, \
         patch("app.core.database.ocr_results_collection") as mock_ocr, \
         patch("app.core.database.trf_data_collection") as mock_trf:
        
        # Configure document collection mock
        mock_docs.find_one.return_value = {
            "id": "test_document_id",
            "file_name": "test.pdf",
            "file_path": "/tmp/test.pdf",
            "file_type": "pdf",
            "status": "processed",
            "ocr_result_id": "test_ocr_result_id",
            "trf_data_id": "test_trf_data_id"
        }
        
        # Configure OCR result collection mock
        mock_ocr.find_one.return_value = {
            "id": "test_ocr_result_id",
            "document_id": "test_document_id",
            "text": SAMPLE_OCR_TEXT,
            "confidence": 0.85,
            "processing_time": 1.2,
            "pages": [
                {
                    "page_num": 1,
                    "text": SAMPLE_OCR_TEXT,
                    "blocks": []
                }
            ]
        }
        
        # Configure TRF data collection mock
        mock_trf.find_one.return_value = {
            "id": "test_trf_data_id",
            "document_id": "test_document_id",
            "ocr_result_id": "test_ocr_result_id",
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
            "missing_required_fields": [],
            "low_confidence_fields": [],
            "extracted_fields": {
                "patientInformation.patientName.firstName": 0.9,
                "patientInformation.patientName.lastName": 0.9,
                "patientInformation.gender": 0.8,
                "patientInformation.dob": 0.8,
                "patientInformation.patientInformationPhoneNumber": 0.7,
                "patientInformation.email": 0.9,
                "clinicalSummary.primaryDiagnosis": 0.8,
                "clinicalSummary.diagnosisDate": 0.8
            }
        }
        
        yield mock_docs, mock_ocr, mock_trf


# Mock agent reasoning
@pytest.fixture
def mock_agent_reasoning():
    """Mock agent reasoning."""
    with patch.object(AgentReasoning, "query_agent") as mock_query, \
         patch.object(AgentReasoning, "analyze_ocr_result") as mock_analyze, \
         patch.object(AgentReasoning, "suggest_field_value") as mock_suggest:
        
        # Configure query_agent mock
        mock_query.return_value = {
            "query": "What is the patient's age?",
            "response": "Based on the information in the document, the patient's date of birth is 01/15/1980. If we calculate the age as of the current date, the patient would be approximately 43 years old.",
            "suggested_actions": [
                {
                    "type": "update_field",
                    "field_path": "patientInformation.age",
                    "value": "43",
                    "confidence": 0.85,
                    "reasoning": "Calculated from the date of birth (01/15/1980)"
                }
            ],
            "timestamp": 1650000000.0
        }
        
        # Configure analyze_ocr_result mock
        mock_analyze.return_value = {
            "missing_fields": [],
            "low_confidence_fields": ["patientInformation.patientInformationPhoneNumber"],
            "suggestions": [
                {
                    "field_path": "patientInformation.age",
                    "suggested_value": "43",
                    "confidence": 0.85,
                    "reasoning": "Calculated from the date of birth (01/15/1980)"
                }
            ],
            "completion_percentage": 0.95,
            "analysis_time": 1650000000.0
        }
        
        # Configure suggest_field_value mock
        mock_suggest.return_value = {
            "field_path": "patientInformation.age",
            "suggested_value": "43",
            "confidence": 0.85,
            "reasoning": "Calculated from the date of birth (01/15/1980)",
            "timestamp": 1650000000.0
        }
        
        yield mock_query, mock_analyze, mock_suggest


# Mock agent suggestions
@pytest.fixture
def mock_agent_suggestions():
    """Mock agent suggestions."""
    with patch.object(AgentSuggestions, "generate_suggestions") as mock_generate, \
         patch.object(AgentSuggestions, "get_field_suggestions") as mock_field, \
         patch.object(AgentSuggestions, "get_missing_field_suggestions") as mock_missing, \
         patch.object(AgentSuggestions, "get_completion_guidance") as mock_guidance:
        
        # Configure generate_suggestions mock
        mock_generate.return_value = {
            "document_id": "test_document_id",
            "ocr_result_id": "test_ocr_result_id",
            "suggestions": [
                {
                    "field_path": "patientInformation.age",
                    "suggested_value": "43",
                    "confidence": 0.85,
                    "reasoning": "Calculated from the date of birth (01/15/1980)"
                }
            ],
            "missing_fields": [],
            "low_confidence_fields": ["patientInformation.patientInformationPhoneNumber"],
            "completion_percentage": 0.95,
            "timestamp": 1650000000.0
        }
        
        # Configure get_field_suggestions mock
        mock_field.return_value = {
            "field_path": "patientInformation.age",
            "suggested_value": "43",
            "confidence": 0.85,
            "reasoning": "Calculated from the date of birth (01/15/1980)",
            "field_description": "Patient's age in years",
            "current_value": None,
            "timestamp": 1650000000.0
        }
        
        # Configure get_missing_field_suggestions mock
        mock_missing.return_value = {
            "missing_fields": [],
            "suggestions": [],
            "timestamp": 1650000000.0
        }
        
        # Configure get_completion_guidance mock
        mock_guidance.return_value = {
            "completion_percentage": 0.95,
            "missing_fields": [],
            "guidance_message": "All required fields are completed. The TRF is ready for submission.",
            "timestamp": 1650000000.0
        }
        
        yield mock_generate, mock_field, mock_missing, mock_guidance


# Test agent query
def test_query_agent(mock_db_connection, mock_collections, mock_agent_reasoning):
    """Test query agent endpoint."""
    # Mock objects
    mock_query, _, _ = mock_agent_reasoning
    
    # Send request
    response = client.post(
        "/api/agent/query/test_document_id",
        json={
            "query": "What is the patient's age?",
            "context": {}
        }
    )
    
    # Check response
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["document_id"] == "test_document_id"
    assert response.json()["query"] == "What is the patient's age?"
    assert "response" in response.json()
    assert "suggested_actions" in response.json()
    
    # Verify agent reasoning was called
    mock_query.assert_called_once()


# Test get suggestions
def test_get_suggestions(mock_db_connection, mock_collections, mock_agent_suggestions):
    """Test get suggestions endpoint."""
    # Mock objects
    mock_generate, _, _, _ = mock_agent_suggestions
    
    # Send request
    response = client.get("/api/agent/suggestions/test_document_id")
    
    # Check response
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["document_id"] == "test_document_id"
    assert "suggestions" in response.json()
    assert "missing_fields" in response.json()
    assert "low_confidence_fields" in response.json()
    assert "completion_percentage" in response.json()
    
    # Verify agent suggestions was called
    mock_generate.assert_called_once()


# Test get field suggestion
def test_get_field_suggestion(mock_db_connection, mock_collections, mock_agent_suggestions):
    """Test get field suggestion endpoint."""
    # Mock objects
    _, mock_field, _, _ = mock_agent_suggestions
    
    # Send request
    response = client.get(
        "/api/agent/suggestions/test_document_id/field",
        params={"field_path": "patientInformation.age"}
    )
    
    # Check response
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["document_id"] == "test_document_id"
    assert response.json()["field_path"] == "patientInformation.age"
    assert "suggestion" in response.json()
    assert response.json()["suggestion"]["suggested_value"] == "43"
    
    # Verify agent suggestions was called
    mock_field.assert_called_once()


# Test get completion guidance
def test_get_completion_guidance(mock_db_connection, mock_collections, mock_agent_suggestions):
    """Test get completion guidance endpoint."""
    # Mock objects
    _, _, _, mock_guidance = mock_agent_suggestions
    
    # Send request
    response = client.get("/api/agent/completion/test_document_id")
    
    # Check response
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["document_id"] == "test_document_id"
    assert "completion_percentage" in response.json()
    assert "missing_fields" in response.json()
    assert "guidance_message" in response.json()
    
    # Verify agent suggestions was called
    mock_guidance.assert_called_once()


# Test get missing field suggestions
def test_get_missing_field_suggestions(mock_db_connection, mock_collections, mock_agent_suggestions):
    """Test get missing field suggestions endpoint."""
    # Mock objects
    _, _, mock_missing, _ = mock_agent_suggestions
    
    # Send request
    response = client.get("/api/agent/suggestions/test_document_id/missing")
    
    # Check response
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["document_id"] == "test_document_id"
    assert "missing_fields" in response.json()
    assert "suggestions" in response.json()
    
    # Verify agent suggestions was called
    mock_missing.assert_called_once()

"""Tests for field extraction functionality."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.models.document import OCRResult
from app.core.field_extractor import FieldExtractor


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
            "status": "ocr_processed",
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
            "physician": {
                "physicianName": "Dr. Jane Johnson",
                "physicianEmail": "jane.johnson@hospital.com",
                "physicianPhoneNumber": "(555) 987-6543"
            },
            "hospital": {
                "hospitalName": "Memorial Hospital",
                "hospitalAddress": "123 Main St, Anytown, CA 12345"
            },
            "Sample": [
                {
                    "sampleType": "Blood",
                    "sampleID": "S12345",
                    "sampleCollectionDate": "06/01/2022"
                }
            ],
            "extraction_confidence": 0.78,
            "missing_required_fields": [],
            "low_confidence_fields": []
        }
        
        yield mock_docs, mock_ocr, mock_trf


# Mock field extractor
@pytest.fixture
def mock_field_extractor():
    """Mock field extractor."""
    with patch.object(FieldExtractor, "extract_fields") as mock_extract:
        mock_extract.return_value = (
            {
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
                }
            },
            {
                "patientInformation.patientName.firstName": 0.9,
                "patientInformation.patientName.lastName": 0.9,
                "patientInformation.gender": 0.8,
                "patientInformation.dob": 0.8,
                "patientInformation.patientInformationPhoneNumber": 0.7,
                "patientInformation.email": 0.9,
                "clinicalSummary.primaryDiagnosis": 0.8,
                "clinicalSummary.diagnosisDate": 0.8
            },
            {
                "total_fields": 8,
                "extracted_fields": 8,
                "high_confidence_fields": 7,
                "low_confidence_fields": 1,
                "extraction_time": 0.5
            }
        )
        yield mock_extract


# Test get TRF data
def test_get_trf_data(mock_db_connection, mock_collections):
    """Test get TRF data endpoint."""
    # Send request
    response = client.get("/api/documents/trf/test_document_id")
    
    # Check response
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["document_id"] == "test_document_id"
    assert response.json()["trf_data_id"] == "test_trf_data_id"
    assert response.json()["trf_data"]["patientInformation"]["patientName"]["firstName"] == "John"
    assert response.json()["trf_data"]["patientInformation"]["patientName"]["lastName"] == "Smith"
    assert response.json()["trf_data"]["clinicalSummary"]["primaryDiagnosis"] == "Breast Cancer"
    
    # Verify database interactions
    mock_docs, _, mock_trf = mock_collections
    mock_docs.find_one.assert_called_once()
    mock_trf.find_one.assert_called_once()


# Test update TRF field
def test_update_trf_field(mock_db_connection, mock_collections):
    """Test update TRF field endpoint."""
    # Send request
    response = client.put(
        "/api/documents/trf/test_document_id/field",
        params={
            "field_path": "patientInformation.patientName.firstName",
            "field_value": "Jonathan",
            "confidence": 0.95
        }
    )
    
    # Check response
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["document_id"] == "test_document_id"
    assert response.json()["field_path"] == "patientInformation.patientName.firstName"
    assert response.json()["previous_value"] == "John"
    assert response.json()["new_value"] == "Jonathan"
    
    # Verify database interactions
    mock_docs, _, mock_trf = mock_collections
    mock_docs.find_one.assert_called_once()
    mock_trf.find_one.assert_called_once()
    mock_trf.update_one.assert_called_once()

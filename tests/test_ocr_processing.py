"""Tests for OCR processing functionality."""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.models.document import OCRResult
from app.core.ocr_service import OCRService


# Create test client
client = TestClient(app)


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
         patch("app.core.database.ocr_results_collection") as mock_ocr:
        
        # Configure mocks
        mock_docs.find_one.return_value = {
            "id": "test_document_id",
            "file_name": "test.pdf",
            "file_path": "/tmp/test.pdf",
            "file_type": "pdf",
            "status": "uploaded",
            "ocr_result_id": "test_ocr_result_id"
        }
        
        mock_ocr.find_one.return_value = {
            "id": "test_ocr_result_id",
            "document_id": "test_document_id",
            "text": "Sample OCR text for testing purposes.",
            "confidence": 0.85,
            "processing_time": 1.2,
            "pages": [
                {
                    "page_num": 1,
                    "text": "Sample OCR text for testing purposes.",
                    "blocks": [
                        {
                            "text": "Sample OCR text",
                            "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.2},
                            "confidence": 0.9,
                            "type": "text"
                        },
                        {
                            "text": "for testing purposes.",
                            "bbox": {"x1": 0.1, "y1": 0.3, "x2": 0.5, "y2": 0.4},
                            "confidence": 0.8,
                            "type": "text"
                        }
                    ]
                }
            ]
        }
        
        yield mock_docs, mock_ocr


# Mock Mistral OCR service
@pytest.fixture
def mock_ocr_service():
    """Mock OCR service."""
    with patch.object(OCRService, "process_document") as mock_process:
        ocr_result = OCRResult(
            document_id="test_document_id",
            text="Sample OCR text for testing purposes.",
            confidence=0.85,
            processing_time=1.2,
            pages=[
                {
                    "page_num": 1,
                    "text": "Sample OCR text for testing purposes.",
                    "blocks": [
                        {
                            "text": "Sample OCR text",
                            "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.2},
                            "confidence": 0.9,
                            "type": "text"
                        },
                        {
                            "text": "for testing purposes.",
                            "bbox": {"x1": 0.1, "y1": 0.3, "x2": 0.5, "y2": 0.4},
                            "confidence": 0.8,
                            "type": "text"
                        }
                    ]
                }
            ]
        )
        mock_process.return_value = ocr_result
        yield mock_process


# Test document processing
def test_process_document(mock_db_connection, mock_collections, mock_ocr_service):
    """Test document processing endpoint."""
    # Send request
    response = client.post(
        "/api/documents/process/test_document_id",
        json={"force_reprocess": False}
    )
    
    # Check response
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["document_id"] == "test_document_id"
    
    # Verify database interactions
    mock_docs, mock_ocr = mock_collections
    mock_docs.find_one.assert_called_once()
    mock_docs.update_one.assert_called()


# Test get OCR result
def test_get_ocr_result(mock_db_connection, mock_collections):
    """Test get OCR result endpoint."""
    # Send request
    response = client.get("/api/documents/ocr/test_document_id")
    
    # Check response
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["document_id"] == "test_document_id"
    assert response.json()["ocr_result_id"] == "test_ocr_result_id"
    assert "text_sample" in response.json()
    assert response.json()["confidence"] == 0.85
    
    # Verify database interactions
    mock_docs, mock_ocr = mock_collections
    mock_docs.find_one.assert_called_once()
    mock_ocr.find_one.assert_called_once()


# Test get OCR result for nonexistent document
def test_get_ocr_result_nonexistent(mock_db_connection, mock_collections):
    """Test get OCR result for nonexistent document."""
    # Configure mock to return None (document not found)
    mock_docs, _ = mock_collections
    mock_docs.find_one.return_value = None
    
    # Send request
    response = client.get("/api/documents/ocr/nonexistent_id")
    
    # Check response
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

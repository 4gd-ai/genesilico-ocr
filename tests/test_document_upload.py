"""Tests for document upload functionality."""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.config import settings


# Create test client
client = TestClient(app)


# Mock database connections
@pytest.fixture(autouse=True)
def mock_db_connection():
    """Mock MongoDB connection."""
    with patch("app.core.database.connect_to_mongodb") as mock_connect:
        mock_connect.return_value = True
        yield mock_connect


# Mock document collection
@pytest.fixture
def mock_documents_collection():
    """Mock documents collection."""
    with patch("app.core.database.documents_collection") as mock_collection:
        mock_collection.insert_one.return_value = MagicMock(inserted_id="test_id")
        mock_collection.find_one.return_value = {
            "id": "test_document_id",
            "file_name": "test.pdf",
            "file_path": "/tmp/test.pdf",
            "file_size": 1024,
            "file_type": "pdf",
            "status": "uploaded"
        }
        yield mock_collection


# Test document upload
def test_upload_document(mock_db_connection, mock_documents_collection):
    """Test document upload endpoint."""
    # Create test file
    test_file_path = "/tmp/test.pdf"
    with open(test_file_path, "wb") as f:
        f.write(b"Test PDF content")
    
    try:
        # Prepare test file
        with open(test_file_path, "rb") as f:
            # Send request
            response = client.post(
                "/api/documents/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"description": "Test document", "auto_process": "false"}
            )
        
        # Check response
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "document_id" in response.json()
        assert response.json()["file_name"] == "test.pdf"
        
        # Verify collection was called
        mock_documents_collection.insert_one.assert_called_once()
        
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)


# Test document upload with invalid file type
def test_upload_document_invalid_type(mock_db_connection):
    """Test document upload with invalid file type."""
    # Create test file
    test_file_path = "/tmp/test.txt"
    with open(test_file_path, "wb") as f:
        f.write(b"Test text content")
    
    try:
        # Prepare test file
        with open(test_file_path, "rb") as f:
            # Send request
            response = client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={"description": "Test document", "auto_process": "false"}
            )
        
        # Check response
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
        
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)


# Test document status
def test_get_document_status(mock_db_connection, mock_documents_collection):
    """Test get document status endpoint."""
    # Send request
    response = client.get("/api/documents/status/test_document_id")
    
    # Check response
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["document_id"] == "test_document_id"
    
    # Verify collection was called
    mock_documents_collection.find_one.assert_called_once()

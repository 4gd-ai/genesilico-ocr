"""Tests for document processor functionality."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from app.core.document_processor import DocumentProcessor
from app.models.document import Document, OCRResult, ProcessingStatus
from app.utils.mongo_helpers import sanitize_mongodb_document


# Sample document data for testing
SAMPLE_DOCUMENT = {
    "id": "test_document_id",
    "file_name": "test.pdf",
    "file_path": "/tmp/test.pdf",
    "file_size": 1024,
    "file_type": "pdf",
    "upload_time": datetime.now(),
    "status": "uploaded",
    "ocr_result_id": None,
    "trf_data_id": None
}

# Sample OCR result data for testing
SAMPLE_OCR_RESULT = {
    "id": "test_ocr_result_id",
    "document_id": "test_document_id",
    "text": "Sample OCR text for testing purposes.",
    "confidence": 0.85,
    "processing_time": 1.2,
    "pages": [
        {
            "page_num": 1,
            "text": "Sample OCR text for testing purposes.",
            "blocks": []
        }
    ]
}

# Sample TRF data for testing
SAMPLE_TRF_DATA = {
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
    }
}

# Sample extraction result for testing
SAMPLE_EXTRACTION_RESULT = (
    {
        "patientID": "TEMP-123456789",
        "patientInformation": {
            "patientName": {
                "firstName": "John",
                "lastName": "Smith"
            }
        },
        "Sample": [{"sampleType": "Blood"}]
    },
    {
        "patientInformation.patientName.firstName": 0.9,
        "patientInformation.patientName.lastName": 0.9
    },
    {
        "total_fields": 2,
        "extracted_fields": 2,
        "high_confidence_fields": 2,
        "low_confidence_fields": 0,
        "extraction_time": 0.8
    }
)


# Mock collections
@pytest.fixture
def mock_collections():
    """Mock database collections properly with AsyncMock for awaitables."""
    with patch("app.core.document_processor.documents_collection") as mock_docs, \
         patch("app.core.document_processor.ocr_results_collection") as mock_ocr, \
         patch("app.core.document_processor.trf_data_collection") as mock_trf, \
         patch("app.core.document_processor.document_groups_collection") as mock_groups, \
         patch("app.core.document_processor.patientreports_collection") as mock_patients:

        # Configure documents collection
        mock_docs.find_one = AsyncMock(return_value=SAMPLE_DOCUMENT)
        mock_docs.update_one = AsyncMock()
        mock_docs.count_documents = AsyncMock(return_value=2)
        mock_docs.find.return_value.skip.return_value.limit.return_value.__aiter__.return_value = [
            {"id": "doc1", "file_name": "doc1.pdf", "status": "processed"},
            {"id": "doc2", "file_name": "doc2.pdf", "status": "processed"},
        ]

        # ✅ Configure OCR results collection (fixes missing await issue)
        mock_ocr.insert_one = AsyncMock()
        mock_ocr.find_one = AsyncMock()

        # Configure TRF data collection
        mock_trf.find_one = AsyncMock(return_value=SAMPLE_TRF_DATA)
        mock_trf.insert_one = AsyncMock()
        mock_trf.update_one = AsyncMock()

        # Configure document groups
        mock_groups.find_one = AsyncMock(return_value={
            "id": "test_group_id",
            "name": "Test Group",
            "document_ids": ["test_document_id1", "test_document_id2"],
            "status": "created"
        })
        mock_groups.update_one = AsyncMock()

        # Configure patient reports
        mock_patients.find_one = AsyncMock(return_value={
            "patientID": "patient-123",
            "patientInformation": {
                "patientName": {
                    "firstName": "John",
                    "lastName": "Smith"
                }
            }
        })
        mock_patients.update_one = AsyncMock()

        yield mock_docs, mock_ocr, mock_trf, mock_groups, mock_patients

# Mock OCR service
@pytest.fixture
def mock_ocr_service():
    """Mock OCR service."""
    with patch("app.core.document_processor.ocr_service") as mock_ocr:
        # Configure OCR service mock
        ocr_result = OCRResult(
            document_id="test_document_id",
            text="Sample OCR text for testing purposes.",
            confidence=0.85,
            processing_time=1.2,
            pages=[
                {
                    "page_num": 1,
                    "text": "Sample OCR text for testing purposes.",
                    "blocks": []
                }
            ]
        )
        mock_ocr.process_document = AsyncMock(return_value=ocr_result)
        yield mock_ocr


# Mock AI field extractor
@pytest.fixture
def mock_ai_field_extractor():
    """Mock AI field extractor."""
    with patch("app.core.document_processor.AIFieldExtractor") as mock_extractor_class:
        # Configure field extractor mock
        mock_extractor = MagicMock()
        mock_extractor.extract_fields = AsyncMock(return_value=SAMPLE_EXTRACTION_RESULT)
        mock_extractor_class.return_value = mock_extractor
        
        # Required for get_trf_data to work in the test
        mock_extractor.get_field_confidence = MagicMock(return_value=0.85)
        
        yield mock_extractor_class


# Test document status retrieval
@pytest.mark.asyncio
async def test_get_document_status(mock_collections):
    """Test retrieving document status."""
    # Get document status
    status = await DocumentProcessor.get_document_status("test_document_id")
    
    # Verify the result
    assert "document_id" in status
    assert status["document_id"] == "test_document_id"
    assert "status" in status
    assert status["status"] == "uploaded"
    assert "file_name" in status
    assert status["file_name"] == "test.pdf"
    
    # Verify database interactions
    mock_docs, mock_ocr, _, _, _ = mock_collections
    mock_docs.find_one.assert_called_once_with({"id": "test_document_id"})


# Test document processing
@pytest.mark.asyncio
async def test_process_document(mock_collections, mock_ocr_service, mock_ai_field_extractor):
    """Test processing a document."""
    # Process document
    result = await DocumentProcessor.process_document("test_document_id")
    
    # Verify the result
    assert "document_id" in result
    assert result["document_id"] == "test_document_id"
    assert "status" in result
    assert result["status"] == "completed"  # This should match what's expected
    assert "ocr_result_id" in result
    assert "trf_data_id" in result
    
    # Verify database interactions
    mock_docs, mock_ocr, mock_trf, _, _ = mock_collections
    mock_docs.find_one.assert_called_once_with({"id": "test_document_id"})
    mock_docs.update_one.assert_called()
    mock_ocr.insert_one.assert_called_once()
    mock_trf.insert_one.assert_called_once()
    
    # Verify OCR service interaction
    mock_ocr_service.process_document.assert_called_once()
    
    # Verify field extractor interaction
    mock_ai_field_extractor.assert_called_once()


# Test document processing for already processed document
@pytest.mark.asyncio
async def test_process_document_already_processed(mock_collections, mock_ocr_service, mock_ai_field_extractor):
    """Test processing an already processed document."""
    # Configure document to be already processed
    mock_docs, _, _, _, _ = mock_collections
    mock_docs.find_one = AsyncMock(return_value={
        **SAMPLE_DOCUMENT,
        "status": "processed"
    })
    
    # Process document without forcing reprocessing
    result = await DocumentProcessor.process_document("test_document_id", force_reprocess=False)
    
    # Verify the result
    assert "message" in result
    assert "already processed" in result["message"]
    assert result["status"] == "completed"
    
    # Verify database interactions
    mock_docs.find_one.assert_called_once_with({"id": "test_document_id"})
    mock_docs.update_one.assert_not_called()
    
    # Verify OCR service interaction
    mock_ocr_service.process_document.assert_not_called()
    
    # Verify field extractor interaction
    mock_ai_field_extractor.assert_not_called()


# Test document processing with forced reprocessing
@pytest.mark.asyncio
async def test_process_document_force_reprocess(mock_collections, mock_ocr_service, mock_ai_field_extractor):
    """Test processing an already processed document with forced reprocessing."""
    # Configure document to be already processed
    mock_docs, _, _, _, _ = mock_collections
    mock_docs.find_one = AsyncMock(return_value={
        **SAMPLE_DOCUMENT,
        "status": "processed"
    })
    
    # Process document with forced reprocessing
    result = await DocumentProcessor.process_document("test_document_id", force_reprocess=True)
    
    # Verify the result
    assert "document_id" in result
    assert result["document_id"] == "test_document_id"
    assert "status" in result
    assert result["status"] == "completed"
    
    # Verify database interactions
    mock_docs.find_one.assert_called_once_with({"id": "test_document_id"})
    mock_docs.update_one.assert_called()
    
    # Verify OCR service interaction
    mock_ocr_service.process_document.assert_called_once()
    
    # Verify field extractor interaction
    mock_ai_field_extractor.assert_called_once()


# Test document processing with patient context
@pytest.mark.asyncio
async def test_process_document_with_patient_context(mock_collections, mock_ocr_service, mock_ai_field_extractor):
    """Test processing a document with patient context."""
    # Configure patient collection to return a patient
    mock_docs, _, _, _, mock_patients = mock_collections
    mock_patients.find_one = AsyncMock(return_value={
        "patientID": "test_patient_id",
        "patientInformation": {
            "patientName": {
                "firstName": "John",
                "lastName": "Smith"
            }
        }
    })
    
    # Process document with patient context
    options = {
        "patient_id": "test_patient_id",
        "save_to_patient_reports": True
    }
    result = await DocumentProcessor.process_document("test_document_id", options=options)
    
    # Verify the result
    assert "document_id" in result
    assert result["document_id"] == "test_document_id"
    assert "status" in result
    assert result["status"] == "completed"
    assert "patient_id" in result
    assert result["patient_id"] == "test_patient_id"
    
    # Verify database interactions
    mock_docs.find_one.assert_called_once_with({"id": "test_document_id"})
    mock_patients.find_one.assert_called_once_with({"patientID": "test_patient_id"})
    
    # Verify field extractor interaction
    mock_ai_field_extractor.assert_called_once()
    args, kwargs = mock_ai_field_extractor.call_args
    assert "existing_patient_data" in kwargs
    assert kwargs["existing_patient_data"] is not None


# Test document group processing
@pytest.mark.asyncio
async def test_process_document_group(mock_collections, mock_ocr_service, mock_ai_field_extractor):
    """Test processing a document group."""
    # Process document group
    result = await DocumentProcessor.process_document_group("test_group_id")
    
    # Verify the result
    assert "group_id" in result
    assert result["group_id"] == "test_group_id"
    assert "status" in result
    assert result["status"] == "completed"
    assert "document_count" in result
    assert "ocr_result_id" in result
    assert "trf_data_id" in result
    
    # Verify database interactions
    mock_docs, mock_ocr, mock_trf, mock_groups, _ = mock_collections
    mock_groups.find_one.assert_called_once_with({"id": "test_group_id"})
    mock_docs.find_one.assert_called_once()
    mock_groups.update_one.assert_called()
    mock_ocr.insert_one.assert_called()
    mock_trf.insert_one.assert_called_once()
    
    # Verify OCR service interaction
    mock_ocr_service.process_document.assert_called_once()
    
    # Verify field extractor interaction
    mock_ai_field_extractor.assert_called_once()


# Test document group processing with patient context
@pytest.mark.asyncio
async def test_process_document_group_with_patient(mock_collections, mock_ocr_service, mock_ai_field_extractor):
    """Test processing a document group with patient context."""
    # Configure patient collection to return a patient
    _, _, _, _, mock_patients = mock_collections
    mock_patients.find_one = AsyncMock(return_value={
        "patientID": "test_patient_id",
        "patientInformation": {
            "patientName": {
                "firstName": "John",
                "lastName": "Smith"
            }
        }
    })
    
    # Process document group with patient context
    options = {
        "patient_id": "test_patient_id",
        "save_to_patient_reports": True
    }
    result = await DocumentProcessor.process_document_group("test_group_id", options=options)
    
    # Verify the result
    assert "group_id" in result
    assert result["group_id"] == "test_group_id"
    assert "status" in result
    assert result["status"] == "completed"
    assert "patient_id" in result
    assert result["patient_id"] == "test_patient_id"
    
    # Verify database interactions with patient
    mock_patients.find_one.assert_called_once_with({"patientID": "test_patient_id"})
    
    # Verify field extractor interaction with patient context
    mock_ai_field_extractor.assert_called_once()
    args, kwargs = mock_ai_field_extractor.call_args
    assert "existing_patient_data" in kwargs
    assert kwargs["existing_patient_data"] is not None


# Test get TRF data
@pytest.mark.asyncio
async def test_get_trf_data(mock_collections):
    """Test get TRF data."""
    # Configure document to have TRF data
    mock_docs, _, _, _, _ = mock_collections
    mock_docs.find_one = AsyncMock(return_value={
        **SAMPLE_DOCUMENT,
        "trf_data_id": "test_trf_data_id"
    })
    
    # Get TRF data
    result = await DocumentProcessor.get_trf_data("test_document_id")
    
    # Verify the result
    assert "document_id" in result
    assert result["document_id"] == "test_document_id"
    assert "trf_data_id" in result
    assert result["trf_data_id"] == "test_trf_data_id"
    assert "trf_data" in result
    
    # Verify database interactions
    mock_docs.find_one.assert_called_once_with({"id": "test_document_id"})
    mock_trf = mock_collections[2]
    mock_trf.find_one.assert_called_once_with({"id": "test_trf_data_id"})


# Test get TRF data for document group
@pytest.mark.asyncio
async def test_get_trf_data_group(mock_collections):
    """Test get TRF data for document group."""
    # Configure document collection to return None for document lookup
    mock_docs, _, _, mock_groups, _ = mock_collections
    mock_docs.find_one = AsyncMock(return_value=None)
    
    # Configure group collection to return a group with TRF data
    mock_groups.find_one = AsyncMock(return_value={
        "id": "test_group_id",
        "name": "Test Group",
        "document_ids": ["test_document_id"],
        "status": "processed",
        "trf_data_id": "test_trf_data_id"
    })
    
    # Get TRF data for group
    result = await DocumentProcessor.get_trf_data("test_group_id")
    
    # Verify the result
    assert "group_id" in result
    assert result["group_id"] == "test_group_id"
    assert "trf_data_id" in result
    assert result["trf_data_id"] == "test_trf_data_id"
    assert "trf_data" in result
    
    # Verify database interactions
    mock_docs.find_one.assert_called_once_with({"id": "test_group_id"})
    mock_groups.find_one.assert_called_once_with({"id": "test_group_id"})
    mock_trf = mock_collections[2]
    mock_trf.find_one.assert_called_once_with({"id": "test_trf_data_id"})


# Test update TRF field
@pytest.mark.asyncio
async def test_update_trf_field(mock_collections):
    """Test updating a TRF field."""
    # Configure document to have TRF data
    mock_docs, _, mock_trf, _, _ = mock_collections
    mock_docs.find_one = AsyncMock(return_value={
        **SAMPLE_DOCUMENT,
        "trf_data_id": "test_trf_data_id"
    })
    
    # Update field
    field_path = "patientInformation.patientName.firstName"
    field_value = "Jonathan"
    confidence = 0.95
    
    # Patch get_field_value and set_field_value
    with patch("app.core.document_processor.get_field_value", return_value="John"), \
         patch("app.core.document_processor.set_field_value"):
        
        # Test the function
        result = await DocumentProcessor.update_trf_field("test_document_id", field_path, field_value, confidence)
        
        # Verify the result
        assert "document_id" in result
        assert result["document_id"] == "test_document_id"
        assert "field_path" in result
        assert result["field_path"] == field_path
        assert "previous_value" in result
        assert result["previous_value"] == "John"
        assert "new_value" in result
        assert result["new_value"] == field_value
        assert "confidence" in result
        assert result["confidence"] == confidence
        
        # Verify database interactions
        mock_docs.find_one.assert_called_once_with({"id": "test_document_id"})
        mock_trf.find_one.assert_called_once_with({"id": "test_trf_data_id"})
        mock_trf.update_one.assert_called_once()


# Test list documents
@pytest.mark.asyncio
async def test_list_documents(mock_collections):
    """Test listing documents."""
    # Configure document collection for list operation
    mock_docs, _, _, _, _ = mock_collections
    mock_docs.count_documents = AsyncMock(return_value=2)
    
    # Mock the cursor
    mock_cursor = MagicMock()
    mock_cursor.__aiter__.return_value = [
        {"id": "doc1", "file_name": "doc1.pdf"},
        {"id": "doc2", "file_name": "doc2.pdf"}
    ]
    mock_docs.find.return_value = mock_cursor
    
    # List documents
    result = await DocumentProcessor.list_documents(10, 0)
    
    # Verify the result
    assert "total" in result
    assert result["total"] == 2
    assert "documents" in result
    assert len(result["documents"]) == 2
    assert result["documents"][0]["id"] == "doc1"
    assert result["documents"][1]["id"] == "doc2"
    
    # Verify database interactions
    mock_docs.count_documents.assert_called_once_with({})
    mock_docs.find.assert_called_once_with({}).skip(0).limit(10)


# Test list documents with status filter
@pytest.mark.asyncio
async def test_list_documents_with_status(mock_collections):
    """Test listing documents with status filter."""
    # Configure document collection for list operation with filter
    mock_docs, _, _, _, _ = mock_collections
    mock_docs.count_documents = AsyncMock(return_value=2)
    
    # Mock the cursor
    mock_cursor = MagicMock()
    mock_cursor.skip.return_value.limit.return_value.__aiter__.return_value = [
        {"id": "doc1", "file_name": "doc1.pdf"},
        {"id": "doc2", "file_name": "doc2.pdf"}
    ]
    mock_docs.find.return_value = mock_cursor
    
    # List documents with status filter
    result = await DocumentProcessor.list_documents(10, 0, status="processed")
    
    # Verify the result
    assert "total" in result
    assert result["total"] == 1
    assert "documents" in result
    assert len(result["documents"]) == 1
    assert result["documents"][0]["id"] == "doc1"
    assert result["documents"][0]["status"] == "processed"
    
    # Verify database interactions
    mock_docs.count_documents.assert_called_once_with({"status": "processed"})
    mock_docs.find.assert_called_once_with({"status": "processed"}).skip(0).limit(10)


# Test helper method extract_nested_fields
def test_extract_nested_fields():
    """Test helper method for extracting nested fields."""
    # Test data
    data = {
        "patientInformation": {
            "patientName": {
                "firstName": "John",
                "lastName": "Smith"
            },
            "gender": "Male"
        },
        "Sample": [
            {"sampleType": "Blood"}
        ]
    }
    
    # Extract nested fields
    result = DocumentProcessor.extract_nested_fields(data)
    
    # Verify the result
    assert isinstance(result, dict)
    assert "patientInformation.patientName.firstName" in result
    assert result["patientInformation.patientName.firstName"] == "John"
    assert "patientInformation.patientName.lastName" in result
    assert result["patientInformation.patientName.lastName"] == "Smith"
    assert "patientInformation.gender" in result
    assert result["patientInformation.gender"] == "Male"
    assert "Sample" in result
    assert result["Sample"] == [{"sampleType": "Blood"}]

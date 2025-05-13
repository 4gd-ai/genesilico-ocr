"""Tests for OCR processing functionality."""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, mock_open

from app.models.document import OCRResult
from app.core.ocr_service import OCRService


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


# Mock collections
@pytest.fixture
def mock_collections():
    """Mock database collections."""
    with patch("app.core.database.documents_collection") as mock_docs, \
         patch("app.core.database.ocr_results_collection") as mock_ocr, \
         patch("app.core.database.trf_data_collection") as mock_trf, \
         patch("app.core.database.patientreports_collection") as mock_patients:
        
        # Configure mocks
        mock_docs.find_one = AsyncMock(return_value={
            "id": "test_document_id",
            "file_name": "test.pdf",
            "file_path": "/tmp/test.pdf",
            "file_type": "pdf",
            "status": "uploaded",
            "ocr_result_id": "test_ocr_result_id"
        })
        
        mock_ocr.find_one = AsyncMock(return_value={
            "id": "test_ocr_result_id",
            "document_id": "test_document_id",
            "text": SAMPLE_OCR_TEXT,
            "confidence": 0.85,
            "processing_time": 1.2,
            "pages": [
                {
                    "page_num": 1,
                    "text": SAMPLE_OCR_TEXT,
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
        })
        
        mock_patients.find_one = AsyncMock(return_value={
            "patientID": "patient-123",
            "patientInformation": {
                "patientName": {
                    "firstName": "John",
                    "lastName": "Smith"
                }
            }
        })
        
        yield mock_docs, mock_ocr, mock_trf, mock_patients


# Mock OCR service
@pytest.fixture
def mock_ocr_service():
    """Mock OCR service."""
    with patch.object(OCRService, "process_document") as mock_process:
        ocr_result = OCRResult(
            document_id="test_document_id",
            text=SAMPLE_OCR_TEXT,
            confidence=0.85,
            processing_time=1.2,
            pages=[
                {
                    "page_num": 1,
                    "text": SAMPLE_OCR_TEXT,
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
        mock_process.side_effect = AsyncMock(return_value=ocr_result)
        yield mock_process


# Test OCR service functionality directly
@pytest.mark.asyncio
async def test_ocr_service_process_document(mock_ocr_service):
    """Test OCR service process_document method."""
    # Mock the OCR service
    service = OCRService(api_key="mock_api_key")
    
    # Process a document
    with patch.object(service, "_extract_text_from_pdf") as mock_extract_pdf:
        # Configure the mock to return some text, pages, and confidence
        mock_extract_pdf.side_effect = AsyncMock(return_value=(
            SAMPLE_OCR_TEXT,
            [{"page_num": 1, "text": SAMPLE_OCR_TEXT, "blocks": []}],
            0.9
        ))
        
        # Process a PDF document
        result = await service.process_document("/tmp/test.pdf", "pdf")
        
        # Verify the result
        assert isinstance(result, OCRResult)
        assert result.text == SAMPLE_OCR_TEXT
        assert result.confidence == 0.9
        assert len(result.pages) == 1
        
        # Verify the method was called
        mock_extract_pdf.assert_called_once_with("/tmp/test.pdf")


@pytest.mark.asyncio
async def test_ocr_service_process_image(mock_ocr_service):
    """Test OCR service process_document method with image."""
    # Mock the OCR service
    service = OCRService(api_key="mock_api_key")
    
    # Process an image
    with patch.object(service, "_extract_text_from_image") as mock_extract_img:
        # Configure the mock to return some text, pages, and confidence
        mock_extract_img.side_effect = AsyncMock(return_value=(
            SAMPLE_OCR_TEXT,
            [{"page_num": 1, "text": SAMPLE_OCR_TEXT, "blocks": []}],
            0.85
        ))
        
        # Process an image document
        result = await service.process_document("/tmp/test.jpg", "jpg")
        
        # Verify the result
        assert isinstance(result, OCRResult)
        assert result.text == SAMPLE_OCR_TEXT
        assert result.confidence == 0.85
        assert len(result.pages) == 1
        
        # Verify the method was called
        mock_extract_img.assert_called_once_with("/tmp/test.jpg")


@pytest.mark.asyncio
async def test_ocr_service_unsupported_format():
    """Test OCR service with unsupported file format."""
    # Create OCR service
    service = OCRService(api_key="mock_api_key")
    
    # Try to process an unsupported format
    with pytest.raises(ValueError) as excinfo:
        await service.process_document("/tmp/test.txt", "txt")
    
    # Verify the error message
    assert "Unsupported file type" in str(excinfo.value)


@pytest.mark.asyncio
async def test_ocr_extract_text_from_image():
    """Test OCR service _extract_text_from_image method."""
    # Create OCR service
    service = OCRService(api_key="mock_api_key")
    
    # Mock the genai model
    with patch.object(service.model, "generate_content") as mock_generate:
        # Configure the mock to return a response
        mock_response = MagicMock()
        mock_response.text = SAMPLE_OCR_TEXT
        mock_generate.return_value = mock_response
        
        # Extract text from an image
        with patch("builtins.open", mock_open(read_data=b"test image data")):
            text, pages, confidence = await service._extract_text_from_image("/tmp/test.jpg")
        
        # Verify the result
        assert text == SAMPLE_OCR_TEXT
        assert len(pages) == 1
        assert pages[0]["text"] == SAMPLE_OCR_TEXT
        assert confidence == 1.0
        
        # Verify the method was called
        mock_generate.assert_called_once()


@pytest.mark.asyncio
async def test_ocr_extract_text_from_pdf():
    """Test OCR service _extract_text_from_pdf method."""
    # Create OCR service
    service = OCRService(api_key="mock_api_key")
    
    # Mock pdf2image and extract_text_from_image
    with patch("app.core.ocr_service.pdf2image") as mock_pdf2image, \
         patch.object(service, "_extract_text_from_image") as mock_extract_img:
        # Configure the mocks
        mock_image = MagicMock()
        mock_pdf2image.convert_from_path.return_value = [mock_image, mock_image]  # Two pages
        
        # Mock extract_text_from_image to return text for each page
        mock_extract_img.side_effect = AsyncMock(return_value=(
            "Page text",
            [{"page_num": 1, "text": "Page text", "blocks": []}],
            0.9
        ))
        
        # Mock tempfile.NamedTemporaryFile
        with patch("app.core.ocr_service.tempfile.NamedTemporaryFile") as mock_temp_file:
            # Configure the mock
            mock_context = MagicMock()
            mock_context.__enter__.return_value.name = "/tmp/temp_image.jpg"
            mock_temp_file.return_value = mock_context
            
            # Extract text from a PDF
            text, pages, confidence = await service._extract_text_from_pdf("/tmp/test.pdf")
        
        # Verify the result
        assert text == "Page text"  # The test only expects one page of text now
        assert len(pages) == 2
        assert pages[0]["text"] == "Page text"
        assert pages[1]["text"] == "Page text"
        assert confidence == 1.0
        
        # Verify the methods were called
        mock_pdf2image.convert_from_path.assert_called_once_with("/tmp/test.pdf")
        assert mock_extract_img.call_count == 2


@pytest.mark.asyncio
async def test_ocr_extract_text_from_pdf_error():
    """Test OCR service _extract_text_from_pdf method with conversion error."""
    # Create OCR service
    service = OCRService(api_key="mock_api_key")
    
    # Mock pdf2image to raise an exception
    with patch("app.core.ocr_service.pdf2image") as mock_pdf2image, \
         patch("app.core.ocr_service.PyPDF2", create=True) as mock_pypdf2:
        # Configure pdf2image to raise an exception
        mock_pdf2image.convert_from_path.side_effect = Exception("PDF conversion error")
        
        # Configure PyPDF2 as a fallback
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Extracted text from PyPDF2"
        mock_reader.pages = [mock_page]
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        # Try to extract text with the fallback method
        with patch("builtins.open", mock_open(read_data=b"test pdf data")):
            text, pages, confidence = await service._extract_text_from_pdf("/tmp/test.pdf")
        
        # Verify the result
        assert "Extracted text from PyPDF2" in text
        assert len(pages) == 1
        assert pages[0]["text"] == "Extracted text from PyPDF2"
        assert confidence == 1.0
        
        # Verify that pdf2image was attempted first, then PyPDF2 as fallback
        mock_pdf2image.convert_from_path.assert_called_once_with("/tmp/test.pdf")
        mock_pypdf2.PdfReader.assert_called_once()


def test_ocr_get_text_by_region():
    """Test OCR service get_text_by_region method."""
    # Create OCR service
    service = OCRService(api_key="mock_api_key")
    
    # Create a test OCR result
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
    
    # Get text from specific regions
    text1 = service.get_text_by_region(ocr_result, 0.05, 0.05, 0.55, 0.25)
    text2 = service.get_text_by_region(ocr_result, 0.05, 0.25, 0.55, 0.45)
    text3 = service.get_text_by_region(ocr_result, 0.6, 0.6, 0.9, 0.9)  # Outside any blocks
    
    # Verify the results
    assert text1 == "Sample OCR text"
    assert text2 == "for testing purposes."
    assert text3 == ""  # No text in the specified region


def test_ocr_get_mime_type():
    """Test OCR service _get_mime_type method."""
    # Create OCR service
    service = OCRService(api_key="mock_api_key")
    
    # Test different file extensions
    assert service._get_mime_type("test.jpg") == "image/jpeg"
    assert service._get_mime_type("test.jpeg") == "image/jpeg"
    assert service._get_mime_type("test.png") == "image/png"
    assert service._get_mime_type("test.gif") == "image/gif"
    assert service._get_mime_type("test.webp") == "image/webp"
    assert service._get_mime_type("test.unknown") == "image/jpeg"  # Default

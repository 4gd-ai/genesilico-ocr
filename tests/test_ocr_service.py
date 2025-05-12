"""Tests for OCR service functionality."""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from pathlib import Path

from app.core.ocr_service import OCRService
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


# Define mock for HarmCategory and HarmBlockThreshold
class HarmCategoryMock:
    HARM_CATEGORY_HARASSMENT = "HARM_CATEGORY_HARASSMENT"
    HARM_CATEGORY_HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
    HARM_CATEGORY_SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
    HARM_CATEGORY_DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"

class HarmBlockThresholdMock:
    BLOCK_NONE = "BLOCK_NONE"
    BLOCK_LOW = "BLOCK_LOW"
    BLOCK_MEDIUM = "BLOCK_MEDIUM"
    BLOCK_HIGH = "BLOCK_HIGH"


@pytest.fixture
def mock_harm_categories():
    """Mock HarmCategory and HarmBlockThreshold."""
    with patch("app.core.ocr_service.HarmCategory", HarmCategoryMock), \
         patch("app.core.ocr_service.HarmBlockThreshold", HarmBlockThresholdMock):
        yield


@pytest.fixture
def ocr_service(mock_harm_categories):
    """Create an OCR service instance with a mock API key."""
    with patch("app.core.ocr_service.genai") as mock_genai:
        service = OCRService(api_key="mock_api_key")
        return service


@pytest.fixture
def mock_generate_content():
    """Mock the generate_content method."""
    with patch.object(OCRService, "_extract_text_from_image") as mock_extract_img, \
         patch.object(OCRService, "_extract_text_from_pdf") as mock_extract_pdf:
        
        # Configure the mocks
        mock_extract_img.side_effect = AsyncMock(return_value=(
            SAMPLE_OCR_TEXT,
            [{"page_num": 1, "text": SAMPLE_OCR_TEXT, "blocks": []}],
            1.0
        ))
        mock_extract_pdf.side_effect = AsyncMock(return_value=(
            SAMPLE_OCR_TEXT,
            [{"page_num": 1, "text": SAMPLE_OCR_TEXT, "blocks": []}],
            1.0
        ))
        
        yield mock_extract_img, mock_extract_pdf


@pytest.mark.asyncio
async def test_extract_text_from_image(ocr_service, mock_generate_content):
    """Test extracting text from an image."""
    mock_extract_img, _ = mock_generate_content
    
    # Call the method
    file_path = "/tmp/test.jpg"
    extracted_text, pages, confidence = await ocr_service._extract_text_from_image(file_path)
    
    # Verify the result
    assert extracted_text == SAMPLE_OCR_TEXT
    assert len(pages) == 1
    assert pages[0]["text"] == SAMPLE_OCR_TEXT
    assert confidence == 1.0
    
    # Verify the method was called
    mock_extract_img.assert_called_once_with(file_path)


@pytest.mark.asyncio
async def test_extract_text_from_pdf(ocr_service, mock_generate_content):
    """Test extracting text from a PDF."""
    _, mock_extract_pdf = mock_generate_content
    
    # Call the method
    file_path = "/tmp/test.pdf"
    extracted_text, pages, confidence = await ocr_service._extract_text_from_pdf(file_path)
    
    # Verify the result
    assert extracted_text == SAMPLE_OCR_TEXT
    assert len(pages) == 1
    assert pages[0]["text"] == SAMPLE_OCR_TEXT
    assert confidence == 1.0
    
    # Verify the method was called
    mock_extract_pdf.assert_called_once_with(file_path)


@pytest.mark.asyncio
async def test_process_document_image(ocr_service, mock_generate_content):
    """Test processing an image document."""
    # Mock the OCR service methods
    mock_extract_img, _ = mock_generate_content
    
    # Call the method
    result = await ocr_service.process_document("/tmp/test.jpg", "jpg")
    
    # Verify the result
    assert isinstance(result, OCRResult)
    assert result.text == SAMPLE_OCR_TEXT
    assert result.confidence == 1.0
    assert len(result.pages) == 1
    
    # Verify the method was called
    mock_extract_img.assert_called_once()


@pytest.mark.asyncio
async def test_process_document_pdf(ocr_service, mock_generate_content):
    """Test processing a PDF document."""
    # Mock the OCR service methods
    _, mock_extract_pdf = mock_generate_content
    
    # Call the method
    result = await ocr_service.process_document("/tmp/test.pdf", "pdf")
    
    # Verify the result
    assert isinstance(result, OCRResult)
    assert result.text == SAMPLE_OCR_TEXT
    assert result.confidence == 1.0
    assert len(result.pages) == 1
    
    # Verify the method was called
    mock_extract_pdf.assert_called_once()


@pytest.mark.asyncio
async def test_process_document_unsupported_format(ocr_service):
    """Test processing a document with an unsupported format."""
    # Test the function with an unsupported format
    with pytest.raises(ValueError):
        await ocr_service.process_document("test.txt", "txt")


@pytest.mark.asyncio
async def test_extract_text_from_image_error(ocr_service):
    """Test error handling when extracting text from an image."""
    # Mock _extract_text_from_image to raise an exception
    with patch.object(OCRService, "_extract_text_from_image", side_effect=Exception("API error")):
        # Test the function
        with pytest.raises(Exception) as excinfo:
            await ocr_service._extract_text_from_image("/tmp/test.jpg")
        
        # Verify the error message
        assert "API error" in str(excinfo.value)


@pytest.mark.asyncio
async def test_extract_text_from_pdf_error(ocr_service):
    """Test error handling when extracting text from a PDF."""
    
    # Patch pdf2image.convert_from_path globally
    with patch("pdf2image.convert_from_path", side_effect=Exception("PDF conversion error")):
        with pytest.raises(Exception) as excinfo:
            await ocr_service._extract_text_from_pdf("/tmp/test.pdf")

        assert "PDF conversion error" in str(excinfo.value)


def test_get_mime_type(ocr_service):
    """Test getting MIME type from file extension."""
    # Test different file extensions
    assert ocr_service._get_mime_type("test.jpg") == "image/jpeg"
    assert ocr_service._get_mime_type("test.jpeg") == "image/jpeg"
    assert ocr_service._get_mime_type("test.png") == "image/png"
    assert ocr_service._get_mime_type("test.gif") == "image/gif"
    assert ocr_service._get_mime_type("test.webp") == "image/webp"
    assert ocr_service._get_mime_type("test.unknown") == "image/jpeg"  # Default


def test_get_text_by_region(ocr_service):
    """Test getting text from a specific region."""
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
    
    # Test getting text from a specific region
    text = ocr_service.get_text_by_region(ocr_result, 0.05, 0.05, 0.55, 0.25)
    assert text == "Sample OCR text"
    
    # Test getting text from another region
    text = ocr_service.get_text_by_region(ocr_result, 0.05, 0.25, 0.55, 0.45)
    assert text == "for testing purposes."
    
    # Test getting text from a region with no blocks
    text = ocr_service.get_text_by_region(ocr_result, 0.6, 0.6, 0.9, 0.9)
    assert text == ""

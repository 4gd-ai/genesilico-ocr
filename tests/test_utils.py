"""Tests for utility functions."""

import os
import pytest
import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from app.utils.file_utils import generate_unique_filename, save_uploaded_file, get_file_info, delete_file
from app.utils.ocr_utils import convert_pdf_to_images, optimize_image_for_ocr, get_image_dimensions
from app.utils.normalization import normalize_array_fields
from app.utils.log_utils import get_logger, log_request, log_response, log_document_processing, log_ocr_result, log_field_extraction


# Test file utility functions
class TestFileUtils:
    def test_generate_unique_filename(self):
        """Test generating a unique filename."""
        # Generate unique filename
        original_filename = "test.pdf"
        unique_filename = generate_unique_filename(original_filename)
        
        # Verify the result
        assert isinstance(unique_filename, str)
        assert unique_filename.endswith(".pdf")
        assert len(unique_filename) > len(original_filename)
        
        # Check if it contains a UUID
        file_base = unique_filename.split('.')[0]
        try:
            uuid_obj = uuid.UUID(file_base)
            assert isinstance(uuid_obj, uuid.UUID)
        except ValueError:
            pytest.fail("Filename does not contain a valid UUID")
    
    def test_save_uploaded_file(self):
        """Test saving an uploaded file."""
        # Mock necessary functions
        with patch("builtins.open", mock_open()) as mock_file, \
             patch("os.makedirs") as mock_makedirs, \
             patch("os.path.getsize", return_value=1024) as mock_getsize:
            
            # Save file
            file_data = b"Test file content"
            filename = "test.txt"
            temp_dir = Path("/tmp/uploads")
            
            file_path, file_size = save_uploaded_file(file_data, filename, temp_dir)
            
            # Verify the result
            assert isinstance(file_path, str)
            assert file_path.startswith(str(temp_dir))
            assert file_path.endswith(".txt")
            assert file_size == 1024
            
            # Verify function calls
            mock_makedirs.assert_called_once_with(temp_dir, exist_ok=True)
            mock_file.assert_called_once()
            mock_getsize.assert_called_once()
    
    def test_get_file_info(self):
        """Test getting file information."""
        # Create a mock path for testing
        fake_path = "/tmp/test.txt"
        
        # Mock file operations
        with patch("os.path.isfile", return_value=True), \
             patch("os.path.basename", return_value="test.txt"), \
             patch("os.path.getsize", return_value=1024), \
             patch("os.path.splitext", return_value=("test", ".txt")), \
             patch("os.path.getctime", return_value=1617235200), \
             patch("os.path.getmtime", return_value=1617235300):
            
            # Get file info
            file_info = get_file_info(fake_path)
            
            # Verify the result
            assert isinstance(file_info, dict)
            assert file_info["file_path"] == fake_path
            assert file_info["file_name"] == "test.txt"
            assert file_info["file_size"] == 1024
            assert file_info["file_extension"] == "txt"
            assert file_info["created_time"] == 1617235200
            assert file_info["modified_time"] == 1617235300
    
    def test_get_file_info_nonexistent(self):
        """Test getting file information for nonexistent file."""
        # Mock isfile to return False for nonexistent file
        with patch("os.path.isfile", return_value=False):
            # Try to get file info for nonexistent file
            with pytest.raises(FileNotFoundError):
                get_file_info("/nonexistent/file.txt")
    
    def test_delete_file(self):
        """Test deleting a file."""
        # Mock os.path.isfile and os.remove
        with patch("os.path.isfile", return_value=True), \
             patch("os.remove") as mock_remove:
            
            # Delete file
            result = delete_file("/tmp/test.txt")
            
            # Verify the result
            assert result is True
            mock_remove.assert_called_once_with("/tmp/test.txt")
    
    def test_delete_file_nonexistent(self):
        """Test deleting a nonexistent file."""
        # Mock os.path.isfile to return False
        with patch("os.path.isfile", return_value=False):
            # Try to delete nonexistent file
            result = delete_file("/nonexistent/file.txt")
            
            # Verify the result
            assert result is False


# Test OCR utility functions
class TestOCRUtils:
    def test_convert_pdf_to_images(self):
        """Test converting PDF to images."""
        # Mock pdf2image and temporary directory
        with patch("app.utils.ocr_utils.pdf2image") as mock_pdf2image, \
             patch("app.utils.ocr_utils.tempfile.TemporaryDirectory") as mock_temp_dir:
            
            # Configure mocks
            mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_dir"
            mock_image = MagicMock()
            mock_pdf2image.convert_from_path.return_value = [mock_image, mock_image]  # Two pages
            
            # Convert PDF to images
            result = convert_pdf_to_images("/tmp/test.pdf")
            
            # Verify the result
            assert isinstance(result, list)
            assert len(result) == 2
            assert all(p.startswith("/tmp/test_dir/page_") for p in result)
            
            # Verify pdf2image was called correctly
            mock_pdf2image.convert_from_path.assert_called_once_with(
                "/tmp/test.pdf",
                dpi=300,
                output_folder="/tmp/test_dir",
                fmt="png"
            )
    
    def test_optimize_image_for_ocr(self):
        """Test optimizing an image for OCR."""
        # Mock PIL Image
        with patch("app.utils.ocr_utils.Image") as mock_image_module:
            # Configure mocks
            mock_image = MagicMock()
            mock_image.mode = "RGB"
            mock_image_module.open.return_value = mock_image
            
            # Optimize image
            result = optimize_image_for_ocr("/tmp/test.jpg")
            
            # Verify the result
            assert result == "/tmp/test.jpg"
            
            # Verify Image.open was called correctly
            mock_image_module.open.assert_called_once_with("/tmp/test.jpg")
            
            # Verify image.save was called
            mock_image.save.assert_called_once_with("/tmp/test.jpg", "PNG")
    
    def test_optimize_image_for_ocr_non_rgb(self):
        """Test optimizing a non-RGB image for OCR."""
        # Mock PIL Image
        with patch("app.utils.ocr_utils.Image") as mock_image_module:
            # Configure mocks
            mock_image = MagicMock()
            mock_image.mode = "RGBA"
            mock_rgb_image = MagicMock()
            mock_image.convert.return_value = mock_rgb_image
            mock_image_module.open.return_value = mock_image
            
            # Optimize image
            result = optimize_image_for_ocr("/tmp/test.png")
            
            # Verify the result
            assert result == "/tmp/test.png"
            
            # Verify Image.open and convert were called correctly
            mock_image_module.open.assert_called_once_with("/tmp/test.png")
            mock_image.convert.assert_called_once_with("RGB")
            mock_rgb_image.save.assert_called_once_with("/tmp/test.png", "PNG")
    
    def test_optimize_image_for_ocr_with_output_path(self):
        """Test optimizing an image with a specified output path."""
        # Mock PIL Image
        with patch("app.utils.ocr_utils.Image") as mock_image_module:
            # Configure mocks
            mock_image = MagicMock()
            mock_image.mode = "RGB"
            mock_image_module.open.return_value = mock_image
            
            # Optimize image with output path
            result = optimize_image_for_ocr("/tmp/test.jpg", "/tmp/output.jpg")
            
            # Verify the result
            assert result == "/tmp/output.jpg"
            
            # Verify Image.open was called correctly
            mock_image_module.open.assert_called_once_with("/tmp/test.jpg")
            
            # Verify image.save was called with the output path
            mock_image.save.assert_called_once_with("/tmp/output.jpg", "PNG")
    
    def test_get_image_dimensions(self):
        """Test getting image dimensions."""
        # Mock PIL Image
        with patch("app.utils.ocr_utils.Image") as mock_image_module:
            # Configure mocks
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image_module.open.return_value = mock_image
            
            # Get image dimensions
            width, height = get_image_dimensions("/tmp/test.jpg")
            
            # Verify the result
            assert width == 800
            assert height == 600
            
            # Verify Image.open was called correctly
            mock_image_module.open.assert_called_once_with("/tmp/test.jpg")


# Test normalization functions
class TestNormalization:
    def test_normalize_array_fields_empty(self):
        """Test normalizing array fields with empty data."""
        # Normalize empty data
        data = {}
        result = normalize_array_fields(data)
        
        # Verify the result
        assert result == {}
    
    def test_normalize_array_fields_numeric_dict(self):
        """Test normalizing array fields with numeric dictionary."""
        # Normalize data with numeric dictionary
        data = {
            "Sample": {
                "0": {"sampleType": "Blood"},
                "1": {"sampleType": "Tissue"}
            }
        }
        result = normalize_array_fields(data)
        
        # Verify the result
        assert "Sample" in result
        assert isinstance(result["Sample"], list)
        assert len(result["Sample"]) == 2
        assert result["Sample"][0]["sampleType"] == "Blood"
        assert result["Sample"][1]["sampleType"] == "Tissue"
    
    def test_normalize_array_fields_string(self):
        """Test normalizing array fields with string value."""
        # Normalize data with string value
        data = {
            "Sample": "Blood"
        }
        result = normalize_array_fields(data)
        
        # Verify the result
        assert "Sample" in result
        assert isinstance(result["Sample"], list)
        assert len(result["Sample"]) == 1
        assert result["Sample"][0] == "Blood"
    
    def test_normalize_array_fields_empty_string(self):
        """Test normalizing array fields with empty string value."""
        # Normalize data with empty string value
        data = {
            "Sample": "   "
        }
        result = normalize_array_fields(data)
        
        # Verify the result
        assert "Sample" in result
        assert isinstance(result["Sample"], list)
        assert len(result["Sample"]) == 0
    
    def test_normalize_array_fields_nested(self):
        """Test normalizing nested array fields."""
        # Normalize data with nested array fields
        data = {
            "FamilyHistory": {
                "familyMember": {
                    "0": {"relation": "Father", "condition": "Diabetes"},
                    "1": {"relation": "Mother", "condition": "Hypertension"}
                }
            }
        }
        result = normalize_array_fields(data)
        
        # Verify the result
        assert "FamilyHistory" in result
        assert "familyMember" in result["FamilyHistory"]
        assert isinstance(result["FamilyHistory"]["familyMember"], list)
        assert len(result["FamilyHistory"]["familyMember"]) == 2
        assert result["FamilyHistory"]["familyMember"][0]["relation"] == "Father"
        assert result["FamilyHistory"]["familyMember"][1]["relation"] == "Mother"
    
    def test_normalize_array_fields_dict_to_list(self):
        """Test normalizing Sample when it's a dict instead of a list."""
        # Normalize data with Sample as dict
        data = {
            "Sample": {"sampleType": "Blood"}
        }
        result = normalize_array_fields(data)
        
        # Verify the result
        assert "Sample" in result
        assert isinstance(result["Sample"], list)
        assert len(result["Sample"]) == 1
        assert result["Sample"][0]["sampleType"] == "Blood"


# Test logging functions
class TestLogging:
    def test_get_logger(self):
        """Test getting a logger."""
        # Mock settings and logging
        with patch("app.utils.log_utils.settings") as mock_settings, \
             patch("app.utils.log_utils.logging") as mock_logging, \
             patch("app.utils.log_utils.os") as mock_os:
            # Configure mocks
            mock_settings.LOG_LEVEL = "INFO"
            mock_settings.ENV = "development"
            mock_logger = MagicMock()
            mock_logging.getLogger.return_value = mock_logger
            
            # Get logger
            logger = get_logger("test")
            
            # Verify the result
            assert logger == mock_logger
            
            # Verify logger configuration
            mock_logging.getLogger.assert_called_once_with("test")
            mock_logger.setLevel.assert_called_once_with("INFO")
    
    def test_get_logger_production(self):
        """Test getting a logger in production environment."""
        # Mock settings and logging
        with patch("app.utils.log_utils.settings") as mock_settings, \
             patch("app.utils.log_utils.logging") as mock_logging, \
             patch("app.utils.log_utils.os") as mock_os:
            # Configure mocks
            mock_settings.LOG_LEVEL = "INFO"
            mock_settings.ENV = "production"
            mock_logger = MagicMock()
            mock_logging.getLogger.return_value = mock_logger
            mock_file_handler = MagicMock()
            mock_logging.FileHandler.return_value = mock_file_handler
            mock_formatter = MagicMock()
            mock_logging.Formatter.return_value = mock_formatter
            mock_os.path.dirname.return_value = "/app"
            mock_os.path.join.side_effect = lambda *args: "/".join(args)
            
            # Get logger
            logger = get_logger("test")
            
            # Verify the result
            assert logger == mock_logger
            
            # Verify logger configuration
            mock_logging.getLogger.assert_called_once_with("test")
            mock_logger.setLevel.assert_called_once_with("INFO")
            mock_logging.FileHandler.assert_called_once()
            mock_file_handler.setLevel.assert_called_once_with("INFO")
            mock_file_handler.setFormatter.assert_called_once_with(mock_formatter)
            mock_logger.addHandler.assert_called_once_with(mock_file_handler)
    
    def test_log_request(self):
        """Test logging an API request."""
        # Create mock logger
        mock_logger = MagicMock()
        
        # Log request
        log_request(mock_logger, "req123", "GET", "/api/test", {"param": "value"})
        
        # Verify logging
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "API Request" in log_message
        assert "req123" in log_message
        assert "GET" in log_message
        assert "/api/test" in log_message
        assert "param" in log_message
    
    def test_log_response(self):
        """Test logging an API response."""
        # Create mock logger
        mock_logger = MagicMock()
        
        # Log response
        log_response(mock_logger, "req123", 200, 0.5, {"result": "success"})
        
        # Verify logging
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "API Response" in log_message
        assert "req123" in log_message
        assert "200" in log_message
        assert "0.5" in log_message
        assert "success" in log_message
    
    def test_log_response_truncate_large_text(self):
        """Test logging an API response with large text (truncation)."""
        # Create mock logger
        mock_logger = MagicMock()
        
        # Create large response data
        large_text = "x" * 1000
        response_data = {"text": large_text}
        
        # Log response
        log_response(mock_logger, "req123", 200, 0.5, response_data)
        
        # Verify logging
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "API Response" in log_message
        assert "req123" in log_message
        assert "200" in log_message
        assert "..." in log_message  # Truncated
        assert len(log_message) < len(large_text)
    
    def test_log_document_processing(self):
        """Test logging document processing events."""
        # Create mock logger
        mock_logger = MagicMock()
        
        # Log document processing
        log_document_processing(mock_logger, "doc123", "OCR", "completed", {"confidence": 0.85})
        
        # Verify logging
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Document Processing" in log_message
        assert "doc123" in log_message
        assert "OCR" in log_message
        assert "completed" in log_message
        assert "confidence" in log_message
        assert "0.85" in log_message
    
    def test_log_ocr_result(self):
        """Test logging OCR result events."""
        # Create mock logger
        mock_logger = MagicMock()
        
        # Log OCR result
        log_ocr_result(mock_logger, "doc123", "ocr123", 0.85, 1.2)
        
        # Verify logging
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "OCR Result" in log_message
        assert "doc123" in log_message
        assert "ocr123" in log_message
        assert "0.85" in log_message
        assert "1.2" in log_message
    
    def test_log_field_extraction(self):
        """Test logging field extraction events."""
        # Create mock logger
        mock_logger = MagicMock()
        
        # Log field extraction
        log_field_extraction(mock_logger, "doc123", "trf123", 0.78, 2, 3)
        
        # Verify logging
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Field Extraction" in log_message
        assert "doc123" in log_message
        assert "trf123" in log_message
        assert "0.78" in log_message
        assert "2" in log_message
        assert "3" in log_message

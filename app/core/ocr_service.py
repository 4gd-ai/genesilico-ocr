import os
import time
import base64
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import json

# Import Mistral with error handling
try:
    # Import using the correct module structure per latest documentation
    from mistralai import Mistral
except ImportError:
    # Fallback to a mock implementation for development/testing
    class Mistral:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.ocr = type('', (), {})()
            # Simple mock that returns fixed content (for testing)
            self.ocr.process = lambda **kwargs: type('', (), {
                'pages': [
                    type('', (), {
                        'markdown': 'Mock OCR markdown text for development and testing purposes.',
                        'images': []
                    })
                ],
                'text': 'Mock OCR markdown text for development and testing purposes.',
            })()
        def files(self):
            pass  # In real usage, this method is provided by the library
    print("WARNING: Using mock Mistral OCR implementation for development/testing purposes.")

from PIL import Image
import pdf2image  # (May not be needed if OCR API handles PDFs directly)

from ..config import settings
from ..models.document import OCRResult


class OCRService:
    """Service for OCR processing using the Mistral OCR API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OCR service with the API key."""
        self.api_key = api_key or settings.MISTRAL_API_KEY
        self.client = Mistral(api_key=self.api_key)
    
    def _upload_local_file(self, file_path: str) -> str:
        """
        Upload a local file using the Mistral file API and return a signed URL.
        
        Args:
            file_path: Local file path.
        
        Returns:
            A signed URL string that can be used in the OCR request.
        """
        # Open the file in binary mode.
        filename = os.path.basename(file_path)
        with open(file_path, "rb") as file_obj:
            # Upload the file with purpose "ocr"
            uploaded_file = self.client.files.upload(
                file={
                    "file_name": filename,
                    "content": file_obj,
                },
                purpose="ocr"
            )
        # Retrieve the signed URL for this file.
        signed_url_response = self.client.files.get_signed_url(file_id=uploaded_file.id)
        return signed_url_response.url

    async def _extract_text_from_pdf(self, file_path: str) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Extract text from a PDF document using Mistral OCR API.
        
        Args:
            file_path: Local path to the PDF file.
            
        Returns:
            Tuple of (extracted_text, pages_data, avg_confidence)
        """
        start_time = time.time()

        try:
            # Upload local PDF to obtain a signed URL
            document_url = self._upload_local_file(file_path)
            
            # Process OCR using the remote document URL.
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": document_url
                },
                include_image_base64=True
            )
            
            # Concatenate page markdown into full text.
            all_text = "\n".join(page.markdown for page in ocr_response.pages)
            pages = []
            
            # For each page, build page data using markdown.
            for i, page in enumerate(ocr_response.pages):
                page_data = {
                    "page_num": i + 1,
                    "text": page.markdown,
                    "blocks": []  # Blocks are not provided in the new API response.
                }
                pages.append(page_data)
            
            # The new API does not return confidence values; set a default.
            avg_confidence = 1.0
            
            processing_time = time.time() - start_time
            return all_text, pages, avg_confidence
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise
    
    async def _extract_text_from_image(self, file_path: str) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Extract text from an image using Mistral OCR API.
        
        Args:
            file_path: Local path to the image file.
            
        Returns:
            Tuple of (extracted_text, pages_data, avg_confidence)
        """
        start_time = time.time()
        
        try:
            # Upload the image file to obtain a signed URL.
            image_url = self._upload_local_file(file_path)
            
            # Process OCR using "image_url" document type.
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": image_url
                },
                include_image_base64=True
            )
            
            # For an image, assume a single “page” in the response.
            all_text = "\n".join(page.markdown for page in ocr_response.pages)
            pages = []
            page = ocr_response.pages[0]
            page_data = {
                "page_num": 1,
                "text": page.markdown,
                "blocks": []  # No block details in new response.
            }
            pages.append(page_data)
            
            # Set a default confidence as the new API does not provide it.
            avg_confidence = 1.0
            
            processing_time = time.time() - start_time
            return all_text, pages, avg_confidence
            
        except Exception as e:
            print(f"Error processing image: {e}")
            raise

    async def process_document(self, file_path: str, file_type: str) -> OCRResult:
        """
        Process a document with OCR.
        
        Args:
            file_path: Path to the document file (PDF or image).
            file_type: Type of the document (e.g. 'pdf', 'jpg', 'jpeg').
            
        Returns:
            OCRResult object with extracted text and metadata.
        """
        file_path = str(file_path)  # Ensure file_path is a string
        
        try:
            # Select method based on file type.
            if file_type.lower() == 'pdf':
                extracted_text, pages, confidence = await self._extract_text_from_pdf(file_path)
            elif file_type.lower() in ('jpg', 'jpeg', 'png'):
                extracted_text, pages, confidence = await self._extract_text_from_image(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Here we create the OCRResult.
            # Note: processing_time is set to 0.0; you may update as needed.
            ocr_result = OCRResult(
                document_id="",  # This may be set by the caller.
                text=extracted_text,
                confidence=confidence,
                processing_time=0.0,  # Update if needed.
                pages=pages
            )
            
            return ocr_result
            
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            raise
    
    def get_text_by_region(self, ocr_result: OCRResult, x1: float, y1: float, x2: float, y2: float) -> str:
        """
        Extract text from a specific region of the document.
        
        Args:
            ocr_result: OCR result containing page data.
            x1, y1, x2, y2: Normalized coordinates (0-1) defining the target region.
            
        Returns:
            Extracted text from the region.
            
        Note:
            The current Mistral OCR API returns page markdown without block-level bounding boxes.
            Therefore, region-based extraction is not supported unless you enrich the OCR
            response with additional layout analysis.
        """
        extracted_text = ""
        
        # Iterate over pages and check each block's bbox if available.
        for page in ocr_result.pages:
            for block in page.get("blocks", []):
                bbox = block.get("bbox", {})
                if not bbox:
                    continue
                    
                block_x1 = bbox.get("x1", 0)
                block_y1 = bbox.get("y1", 0)
                block_x2 = bbox.get("x2", 0)
                block_y2 = bbox.get("y2", 0)
                
                # Check if the block overlaps with the given region.
                if (block_x1 < x2 and block_x2 > x1 and
                    block_y1 < y2 and block_y2 > y1):
                    extracted_text += block.get("text", "") + " "
        
        return extracted_text.strip()


# Create a singleton instance
ocr_service = OCRService()

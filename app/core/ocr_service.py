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
    # Try importing from the latest package structure
    from mistralai.client import MistralClient as Mistral
except ImportError:
    try:
        # Try the older package structure
        from mistralai import Mistral
    except ImportError:
        # Fallback to a mock implementation for development/testing
        class Mistral:
            def __init__(self, api_key=None):
                self.api_key = api_key
                self.ocr = type('', (), {})()
                self.ocr.process = lambda **kwargs: type('', (), {
                    'text': 'Mock OCR text for development and testing purposes.',
                    'pages': [
                        type('', (), {
                            'text': 'Mock OCR text for development and testing purposes.',
                            'blocks': [
                                type('', (), {
                                    'text': 'Mock OCR text',
                                    'bbox': {'x1': 0.1, 'y1': 0.1, 'x2': 0.5, 'y2': 0.2},
                                    'confidence': 0.9,
                                    'type': 'text'
                                })
                            ]
                        })
                    ]
                })()
        print("WARNING: Using mock Mistral OCR implementation for development/testing purposes.")

from PIL import Image
import pdf2image

from ..config import settings
from ..models.document import OCRResult


class OCRService:
    """Service for OCR processing using Mistral OCR API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OCR service."""
        self.api_key = api_key or settings.MISTRAL_API_KEY
        self.client = Mistral(api_key=self.api_key)
    
    async def _extract_text_from_pdf(self, file_path: str) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Extract text from a PDF document using Mistral OCR API.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, pages_data, confidence)
        """
        start_time = time.time()
        
        try:
            # Process with Mistral OCR API
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "file_path",
                    "file_path": file_path
                },
                include_image_base64=True
            )
            
            # Extract the full text and page data
            all_text = ocr_response.text
            pages = []
            
            for i, page in enumerate(ocr_response.pages):
                page_data = {
                    "page_num": i + 1,
                    "text": page.text,
                    "blocks": []
                }
                
                for block in page.blocks:
                    block_data = {
                        "text": block.text,
                        "bbox": block.bbox,
                        "confidence": block.confidence,
                        "type": block.type
                    }
                    page_data["blocks"].append(block_data)
                
                pages.append(page_data)
            
            # Calculate average confidence
            block_confidences = []
            for page in ocr_response.pages:
                for block in page.blocks:
                    if hasattr(block, 'confidence') and block.confidence is not None:
                        block_confidences.append(block.confidence)
            
            avg_confidence = sum(block_confidences) / len(block_confidences) if block_confidences else 0.0
            
            processing_time = time.time() - start_time
            return all_text, pages, avg_confidence
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise
    
    async def _extract_text_from_image(self, file_path: str) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Extract text from an image using Mistral OCR API.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Tuple of (extracted_text, pages_data, confidence)
        """
        start_time = time.time()
        
        try:
            # Process with Mistral OCR API
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "file_path",
                    "file_path": file_path
                },
                include_image_base64=True
            )
            
            # For images, we'll have a single page
            all_text = ocr_response.text
            pages = []
            
            # We'll only have one page for an image
            page = ocr_response.pages[0]
            page_data = {
                "page_num": 1,
                "text": page.text,
                "blocks": []
            }
            
            for block in page.blocks:
                block_data = {
                    "text": block.text,
                    "bbox": block.bbox,
                    "confidence": block.confidence,
                    "type": block.type
                }
                page_data["blocks"].append(block_data)
            
            pages.append(page_data)
            
            # Calculate average confidence
            block_confidences = []
            for block in page.blocks:
                if hasattr(block, 'confidence') and block.confidence is not None:
                    block_confidences.append(block.confidence)
            
            avg_confidence = sum(block_confidences) / len(block_confidences) if block_confidences else 0.0
            
            processing_time = time.time() - start_time
            return all_text, pages, avg_confidence
            
        except Exception as e:
            print(f"Error processing image: {e}")
            raise
    
    async def process_document(self, file_path: str, file_type: str) -> OCRResult:
        """
        Process a document with OCR.
        
        Args:
            file_path: Path to the document file
            file_type: Type of the document (pdf, jpg, jpeg)
            
        Returns:
            OCRResult object with extracted text and metadata
        """
        file_path = str(file_path)  # Ensure file_path is a string
        
        try:
            if file_type.lower() in ('pdf'):
                extracted_text, pages, confidence = await self._extract_text_from_pdf(file_path)
            elif file_type.lower() in ('jpg', 'jpeg'):
                extracted_text, pages, confidence = await self._extract_text_from_image(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Create OCR result
            ocr_result = OCRResult(
                document_id="",  # To be set by the caller
                text=extracted_text,
                confidence=confidence,
                processing_time=0.0,  # To be set by the caller
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
            ocr_result: OCR result containing page and block data
            x1, y1, x2, y2: Coordinates of the region (normalized 0-1)
            
        Returns:
            Extracted text from the region
        """
        extracted_text = ""
        
        for page in ocr_result.pages:
            for block in page.get("blocks", []):
                bbox = block.get("bbox", {})
                
                # Check if the block is within the region
                if not bbox:
                    continue
                    
                block_x1 = bbox.get("x1", 0)
                block_y1 = bbox.get("y1", 0)
                block_x2 = bbox.get("x2", 0)
                block_y2 = bbox.get("y2", 0)
                
                # Check for overlap
                if (block_x1 < x2 and block_x2 > x1 and
                    block_y1 < y2 and block_y2 > y1):
                    extracted_text += block.get("text", "") + " "
        
        return extracted_text.strip()


# Create a singleton instance
ocr_service = OCRService()

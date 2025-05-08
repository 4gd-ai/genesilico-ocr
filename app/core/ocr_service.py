import os
import time
import base64
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import json

# Import Gemini with error handling
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    # Fallback to a mock implementation for development/testing
    class GenAIMock:
        def configure(self, **kwargs):
            pass

        class GenerativeModel:
            def __init__(self, *args, **kwargs):
                pass

            def generate_content(self, *args, **kwargs):
                return type('', (), {
                    'text': 'Mock OCR text for development and testing purposes.',
                })()

    genai = GenAIMock()
    print("WARNING: Using mock Gemini OCR implementation for development/testing purposes.")

from PIL import Image
import pdf2image  # Used for converting PDFs to images for Gemini

from ..config import settings
from ..models.document import OCRResult


class OCRService:
    """Service for OCR processing using the Gemini Vision API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OCR service with the API key."""
        self.api_key = api_key or settings.GEMINI_API_KEY
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        # Use Gemini Pro Vision model for OCR
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Configure safety settings (optional)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    def _get_mime_type(self, file_path: str) -> str:
        """
        Determine MIME type based on file extension.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            MIME type as a string.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            return 'image/jpeg'
        elif ext == '.png':
            return 'image/png'
        elif ext == '.gif':
            return 'image/gif'
        elif ext == '.webp':
            return 'image/webp'
        else:
            # Default to jpeg
            return 'image/jpeg'
    
    async def _extract_text_from_image(self, file_path: str) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Extract text from an image using Gemini Vision API.
        
        Args:
            file_path: Local path to the image file.
            
        Returns:
            Tuple of (extracted_text, pages_data, avg_confidence)
        """
        start_time = time.time()
        
        try:
            # Read the image file
            with open(file_path, "rb") as f:
                image_data = f.read()
            
            # Create a prompt for OCR
            prompt = """
            Extract all text from this image, including handwritten text.
            Format it exactly as it appears in the image, preserving line breaks, 
            paragraphs, and text structure. If there are tables, preserve the table structure.
            Return ONLY the extracted text without any additional commentary.
            """
            
            # Generate content with the image
            response = self.model.generate_content(
                [
                    prompt,
                    {"mime_type": self._get_mime_type(file_path), "data": image_data}
                ],
                safety_settings=self.safety_settings
            )
            
            # Extract text from response
            extracted_text = response.text.strip()
            
            # Create a single page with the extracted text
            pages = [{
                "page_num": 1,
                "text": extracted_text,
                "blocks": []  # No block details in Gemini response
            }]
            
            # Gemini doesn't provide confidence scores, set to 1.0
            avg_confidence = 1.0
            
            processing_time = time.time() - start_time
            return extracted_text, pages, avg_confidence
            
        except Exception as e:
            print(f"Error processing image: {e}")
            raise

    # In your ocr_service.py file, modify the _extract_text_from_pdf method:

    async def _extract_text_from_pdf(self, file_path: str) -> Tuple[str, List[Dict[str, Any]], float]:
            """
            Extract text from a PDF document using Gemini API.
            
            Args:
                file_path: Local path to the PDF file.
                
            Returns:
                Tuple of (extracted_text, pages_data, avg_confidence)
            """
            start_time = time.time()

            try:
                # If pdf2image/poppler is available, use it
                try:
                    import pdf2image
                    # Convert PDF to images
                    images = pdf2image.convert_from_path(file_path)
                    
                    all_pages_text = []
                    pages_data = []
                    
                    # Process each page
                    for i, image in enumerate(images):
                        # Save the image to a temporary file
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                            image_path = temp_file.name
                            image.save(image_path, 'JPEG')
                        
                        try:
                            # Extract text from the image
                            page_text, _, _ = await self._extract_text_from_image(image_path)
                            all_pages_text.append(page_text)
                            
                            # Add page data
                            pages_data.append({
                                "page_num": i + 1,
                                "text": page_text,
                                "blocks": []  # No blocks in Gemini response
                            })
                        finally:
                            # Clean up the temporary file
                            if os.path.exists(image_path):
                                os.unlink(image_path)
                    
                    # Combine all page text
                    all_text = "\n\n".join(all_pages_text)
                    
                except (ImportError, Exception) as e:
                    # If pdf2image fails, try directly with PyPDF2 to extract text if possible
                    # or use a direct method that doesn't require conversion
                    print(f"PDF conversion failed: {e}, trying alternative method")
                    
                    # Method 1: Try with PyPDF2 (text-based extraction, won't work with scanned PDFs)
                    try:
                        import PyPDF2
                        
                        all_text = ""
                        pages_data = []
                        
                        with open(file_path, 'rb') as file:
                            pdf_reader = PyPDF2.PdfReader(file)
                            
                            for i, page in enumerate(pdf_reader.pages):
                                page_text = page.extract_text() or ""
                                all_text += page_text + "\n\n"
                                
                                pages_data.append({
                                    "page_num": i + 1,
                                    "text": page_text,
                                    "blocks": []
                                })
                                
                        if not all_text.strip():
                            raise ValueError("No text could be extracted from PDF")
                            
                    except (ImportError, ValueError, Exception):
                        # Method 2: As a last resort, treat the first page as an image
                        # This is not ideal but can work as a fallback
                        print("Trying to process first page of PDF directly as image")
                        
                        # Use PIL to open the first page of the PDF as an image
                        try:
                            from PIL import Image
                            import io
                            
                            # Use the document as is, and let Gemini try to process it
                            all_text, pages_data, _ = await self._extract_text_from_image(file_path)
                            
                        except Exception as e:
                            raise ValueError(f"Could not process PDF with any available method: {e}")
                
                # Gemini doesn't provide confidence scores, set to 1.0
                avg_confidence = 1.0
                
                processing_time = time.time() - start_time
                return all_text, pages_data, avg_confidence
                
            except Exception as e:
                print(f"Error processing PDF: {e}")
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
                    start_time = time.time()
                    
                    # Select method based on file type
                    if file_type.lower() == 'pdf':
                        extracted_text, pages, confidence = await self._extract_text_from_pdf(file_path)
                    elif file_type.lower() in ('jpg', 'jpeg', 'png', 'gif', 'webp'):
                        extracted_text, pages, confidence = await self._extract_text_from_image(file_path)
                    else:
                        raise ValueError(f"Unsupported file type: {file_type}")
                    
                    processing_time = time.time() - start_time
                    
                    # Create the OCRResult
                    ocr_result = OCRResult(
                        document_id="",  # This may be set by the caller
                        text=extracted_text,
                        confidence=confidence,
                        processing_time=processing_time,
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
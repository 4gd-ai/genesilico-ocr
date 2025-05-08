#!/usr/bin/env python
"""Script to test Gemini OCR API for both printed and handwritten text."""

import os
import sys
import argparse
import traceback
from pathlib import Path
import mimetypes
import google.generativeai as genai
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()


def get_mime_type(file_path):
    """Determine MIME type based on file extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type:
        return mime_type
    
    # Fallback to extension-based detection
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
        # Default to jpeg for images
        return 'image/jpeg'


def test_gemini_ocr(api_key, test_file):
    """Test Gemini OCR API for both printed and handwritten text."""
    try:
        # Configure the Gemini API
        print(f"Configuring Gemini client with API key: {api_key[:4]}...{api_key[-4:]}")
        genai.configure(api_key=api_key)
        
        # Use Gemini 2.0 Flash Experimental model for OCR
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Get MIME type
        mime_type = get_mime_type(test_file)
        print(f"Processing file: {test_file} (MIME type: {mime_type})")
        
        # Read the file
        with open(test_file, "rb") as f:
            file_data = f.read()
            file_size = len(file_data)
            print(f"File size: {file_size / 1024:.2f} KB")
        
        # Create a prompt for both printed and handwritten text
        prompt = """
        Extract ALL text from this image, including both printed and handwritten content.
        
        Please transcribe:
        - All printed text in the document
        - All handwritten notes, annotations, or signatures
        - Any text in tables, forms, or structured content
        
        Preserve the exact formatting including:
        - Line breaks and paragraph structure
        - Numbers, dates, and special characters
        - Table structure (if present)
        
        If parts are illegible, indicate with [illegible].
        
        Return ONLY the extracted text without any explanations or commentary.
        """
        
        # Generate content with the image
        print("Sending request to Gemini API...")
        response = model.generate_content(
            [
                prompt,
                {"mime_type": mime_type, "data": file_data}
            ],
            # Lower temperature for more accurate extraction
            generation_config={"temperature": 0.2}
        )
        
        # Print results
        if hasattr(response, 'text') and response.text:
            print("OCR processing successful!")
            extracted_text = response.text
            print(f"Extracted text (first 500 chars): {extracted_text[:500]}...")
            
            # Count lines for basic stats
            lines = [l for l in extracted_text.split('\n') if l.strip()]
            print(f"Number of lines extracted: {len(lines)}")
            
            # Save the extracted text to a file for review
            output_file = f"{os.path.splitext(test_file)[0]}_extracted.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            print(f"Extracted text saved to: {output_file}")
            
            return True
        else:
            print("Warning: Response received but no text was extracted")
            print(f"Full response object: {response}")
            return False
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        traceback.print_exc()
        return False


def main():
    """Test Gemini OCR API."""
    parser = argparse.ArgumentParser(description="Test Gemini OCR API for text extraction")
    parser.add_argument("--api-key", help="Gemini API key", default=os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--file", help="Test file path", required=True)
    args = parser.parse_args()
    
    # Check if API key is provided
    if not args.api_key:
        print("Error: Gemini API key not provided")
        print("Please set the GEMINI_API_KEY environment variable or use the --api-key option")
        return 1
    
    # Check if test file exists
    if not os.path.isfile(args.file):
        print(f"Error: Test file '{args.file}' not found")
        return 1
    
    # Test Gemini OCR API
    success = test_gemini_ocr(args.api_key, args.file)
    
    if success:
        print("Gemini OCR test passed")
        return 0
    else:
        print("Gemini OCR test failed")
        return 1


if __name__ == "__main__":
    exit(main())
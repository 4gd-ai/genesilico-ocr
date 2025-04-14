#!/usr/bin/env python
"""Script to test Mistral OCR API."""

import os
import sys
import argparse
from pathlib import Path
from mistralai import Mistral
from mistralai.exceptions import MistralException
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()


def test_mistral_ocr(api_key, test_file):
    """Test Mistral OCR API."""
    try:
        # Create client
        print(f"Creating Mistral client with API key: {api_key[:4]}...{api_key[-4:]}")
        client = Mistral(api_key=api_key)
        
        # Process test file
        print(f"Processing file: {test_file}")
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "file_path",
                "file_path": test_file
            },
            include_image_base64=False  # Set to False to reduce response size
        )
        
        # Print results
        print("OCR processing successful!")
        print(f"Extracted text (first 500 chars): {ocr_response.text[:500]}...")
        print(f"Number of pages: {len(ocr_response.pages)}")
        
        # Print page information
        for i, page in enumerate(ocr_response.pages):
            print(f"Page {i+1}: {len(page.blocks)} blocks")
            
            # Print first few blocks
            for j, block in enumerate(page.blocks[:3]):
                if hasattr(block, 'text'):
                    print(f"  Block {j+1}: {block.text[:50]}...")
                    if hasattr(block, 'confidence'):
                        print(f"    Confidence: {block.confidence:.2f}")
            
            if len(page.blocks) > 3:
                print(f"  ... and {len(page.blocks) - 3} more blocks")
        
        return True
    except MistralException as e:
        print(f"Mistral API error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def main():
    """Test Mistral OCR API."""
    parser = argparse.ArgumentParser(description="Test Mistral OCR API")
    parser.add_argument("--api-key", help="Mistral API key", default=os.getenv("MISTRAL_API_KEY"))
    parser.add_argument("--file", help="Test file path", required=True)
    args = parser.parse_args()
    
    # Check if API key is provided
    if not args.api_key:
        print("Error: Mistral API key not provided")
        print("Please set the MISTRAL_API_KEY environment variable or use the --api-key option")
        return 1
    
    # Check if test file exists
    if not os.path.isfile(args.file):
        print(f"Error: Test file '{args.file}' not found")
        return 1
    
    # Test Mistral OCR API
    success = test_mistral_ocr(args.api_key, args.file)
    
    if success:
        print("Mistral OCR API test passed")
        return 0
    else:
        print("Mistral OCR API test failed")
        return 1


if __name__ == "__main__":
    exit(main())

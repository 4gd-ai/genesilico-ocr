#!/usr/bin/env python
"""Script to create a sample .env file."""

import os
import uuid
import argparse
from pathlib import Path


def main():
    """Create a sample .env file."""
    parser = argparse.ArgumentParser(description="Create a sample .env file")
    parser.add_argument("--output", "-o", help="Output file path", default=".env")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing file")
    args = parser.parse_args()
    
    # Check if file exists
    if os.path.exists(args.output) and not args.force:
        print(f"Error: File '{args.output}' already exists. Use --force to overwrite.")
        return 1
    
    # Create sample .env content
    env_content = f"""# Application settings
APP_NAME=GenesilicoCRAgent
ENV=development
DEBUG=True
LOG_LEVEL=INFO

# API settings
API_HOST=0.0.0.0
API_PORT=8000

# Database settings
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB=genesilico_ocr

# Mistral AI settings
MISTRAL_API_KEY=your_mistral_api_key_here

# File storage settings
UPLOAD_DIR={Path.cwd() / "data" / "documents"}
OCR_RESULTS_DIR={Path.cwd() / "data" / "ocr_results"}
TRF_OUTPUTS_DIR={Path.cwd() / "data" / "trf_outputs"}
MAX_UPLOAD_SIZE_MB=10
"""
    
    # Write to file
    with open(args.output, "w") as f:
        f.write(env_content)
    
    print(f"Created sample .env file at '{args.output}'")
    print("Remember to update the MISTRAL_API_KEY with your actual API key.")
    return 0


if __name__ == "__main__":
    exit(main())

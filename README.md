# Genesilico OCR + AI Agent Service

A Proof of Concept (POC) for an OCR-powered document processing system with AI agent assistance for extracting and validating data from medical Test Requisition Forms (TRFs).

## Overview

This service implements a pipeline for processing medical documents:
1. Document Intake: API for uploading PDFs and images
2. OCR Processing: Extract text using Mistral OCR
3. Field Extraction: Extract structured data based on TRF schema
4. Schema Validation: Validate extracted data against TRF schema
5. AI Agent: Provide assistance for completing and correcting data

## Features

- Document upload and processing
- OCR text extraction with Mistral OCR
- Automatic field extraction based on patterns and heuristics
- Schema validation for TRF data
- AI agent for field suggestions and data completion
- API for interacting with the service
- Support for processing document groups and merging data

## Tech Stack

- **Backend**: FastAPI + Python
- **OCR**: Mistral OCR API
- **AI**: Mistral AI for the agent
- **Database**: MongoDB
- **Containerization**: Docker + Docker Compose

## Getting Started

### Prerequisites

- Python 3.9 or later
- MongoDB
- Docker and Docker Compose (optional)
- Google Gemini API Key
- Optional: Poppler (for PDF processing with pdf2image)

### Environment Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```
5. Edit the `.env` file to set your API keys and configuration:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   OPENAI_API_KEY=your_openai_api_key
   MONGODB_URL=mongodb://localhost:27017
   MONGODB_DB=genesilico_ocr
   ```

### Running the Application

#### Local Development

```bash
python run.py --env development --reload
```

#### Production Deployment with Docker

```bash
cd docker
docker-compose up -d
```

## API Documentation

Once the service is running, access the API documentation at:
- Swagger UI: `http://localhost:5005/docs`
- ReDoc: `http://localhost:5005/redoc`

### Key Endpoints

- **POST /api/documents/upload**: Upload a document
- **POST /api/documents/process/{document_id}**: Process a document
- **GET /api/documents/status/{document_id}**: Get document processing status
- **GET /api/documents/trf/{document_id}**: Get extracted TRF data
- **POST /api/agent/query/{document_id}**: Query the AI agent
- **GET /api/agent/suggestions/{document_id}**: Get AI agent suggestions

## Project Structure

```
genesilico-ocr/
│
├── app/                        # Main application code
│   ├── api/                    # API endpoints
│   ├── core/                   # Core business logic
│   │   ├── document_processor.py  # Document processing pipeline
│   │   ├── ocr_service.py      # OCR using Gemini Vision API
│   │   ├── field_extractor.py  # AI-based field extraction
│   │   └── schema_validator.py # Data validation
│   ├── agent/                  # AI Agent implementation
│   ├── models/                 # Data models
│   ├── schemas/                # Data schemas
│   └── utils/                  # Utility functions
│
├── data/                       # Data storage
│   ├── documents/              # Uploaded documents
│   ├── ocr_results/            # OCR processing results
│   └── trf_outputs/            # Generated TRF documents
│
├── docker/                     # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── tests/                      # Test suite
│   ├── test_ocr_service.py     # Tests for OCR functionality
│   ├── test_document_processor.py  # Tests for document processing
│   ├── test_field_extraction.py    # Tests for field extraction
│   └── ...                     # Other test files
│
├── .env.example                # Example environment variables
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── run.py                      # Script to run the application
```

## OCR Implementation 

The OCR service uses Google's Gemini Vision API to extract text from documents:

- Supports both image formats (JPG, PNG, GIF, WEBP) and PDFs
- For PDFs, converts pages to images using pdf2image
- Falls back to PyPDF2 for text-based PDFs if pdf2image is unavailable
- Handles multi-page documents and preserves document structure
- Optimized for medical form content, including handwritten text

## Field Extraction 

Text extraction is performed using an AI-based approach:

- Uses GPT-4o to extract structured data from OCR text
- Can incorporate existing patient data as context for improved extraction
- Provides confidence scores for extracted fields
- Identifies missing or low-confidence fields for review

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Fields

To add new fields to the extraction:

1. Update the TRF schema in `app/models/trf.py`
2. Add extraction patterns in `app/schemas/trf_schema.py`
3. Update the field extractor in `app/core/field_extractor.py`

## License

This project is proprietary and confidential.

## Contact

For questions or support, please contact the Genesilico team.

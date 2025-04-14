# Developer Guide for Genesilico OCR + AI Agent Service

This guide provides instructions for developers working on the Genesilico OCR + AI Agent Service.

## Development Environment Setup

### Prerequisites

- Python 3.9 or later
- MongoDB
- Docker and Docker Compose (optional)

### Install Dependencies

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Environment

```bash
# Create a .env file
python scripts/create_env.py

# Edit the .env file to set your Mistral API key
# MISTRAL_API_KEY=your_mistral_api_key_here
```

### Test Setup

```bash
# Test MongoDB connection
python scripts/test_mongodb.py

# Test Mistral OCR API (requires a test file)
python scripts/test_mistral_ocr.py --file path/to/test/file.pdf
```

## Project Structure

```
genesilico-ocr/
│
├── app/                        # Main application code
│   ├── api/                    # API endpoints
│   ├── core/                   # Core business logic
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
├── scripts/                    # Utility scripts
│
├── tests/                      # Test suite
├── .env.example                # Example environment variables
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── run.py                      # Script to run the application
```

## Development Workflow

### Running the Application

```bash
# Development mode with auto-reload
python run.py --env development --reload

# Or using Make
make dev
```

### Running Tests

```bash
pytest tests/

# Or using Make
make test
```

### Docker Development

```bash
# Build Docker image
docker build -t genesilico-ocr -f docker/Dockerfile .

# Run with Docker Compose
cd docker
docker-compose up -d

# Or using Make
make docker-build
make docker-run
```

## Core Components

### Document Processing Pipeline

1. **DocumentProcessor**: Orchestrates the document processing pipeline
2. **OCRService**: Handles OCR processing using Mistral OCR
3. **FieldExtractor**: Extracts structured data from OCR results
4. **SchemaValidator**: Validates extracted data against the TRF schema

### AI Agent

1. **AgentReasoning**: Reasoning engine for analyzing and suggesting improvements
2. **AgentSuggestions**: Generates suggestions for completing and correcting data

## API Endpoints

### Document Management

- **POST /api/documents/upload**: Upload a document
- **POST /api/documents/process/{document_id}**: Process a document
- **GET /api/documents/status/{document_id}**: Get document processing status
- **GET /api/documents/ocr/{document_id}**: Get OCR result
- **GET /api/documents/trf/{document_id}**: Get TRF data
- **PUT /api/documents/trf/{document_id}/field**: Update a field in the TRF data

### AI Agent Interaction

- **POST /api/agent/query/{document_id}**: Query the AI agent
- **GET /api/agent/suggestions/{document_id}**: Get AI agent suggestions
- **GET /api/agent/suggestions/{document_id}/field**: Get field suggestion
- **GET /api/agent/suggestions/{document_id}/missing**: Get suggestions for missing fields
- **GET /api/agent/completion/{document_id}**: Get completion guidance

## Best Practices

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Document functions and classes with docstrings
- Keep functions small and focused on a single responsibility

### Error Handling

- Use try-except blocks to handle exceptions
- Log errors with appropriate context information
- Return meaningful error messages to the client

### Testing

- Write unit tests for all new functionality
- Use mocks to isolate components during testing
- Test both success and failure paths

## Adding New Features

### Extending the TRF Schema

1. Update the TRF model in `app/models/trf.py`
2. Add extraction patterns in `app/schemas/trf_schema.py`
3. Update the field extractor in `app/core/field_extractor.py`

### Adding New API Endpoints

1. Define the route in the appropriate route module
2. Implement the endpoint handler
3. Add request and response schemas
4. Add tests for the new endpoint

## Troubleshooting

### Common Issues

- **MongoDB Connection Errors**: Ensure MongoDB is running and accessible
- **Mistral API Errors**: Check your API key and the API status
- **File Processing Errors**: Verify file formats and permissions

### Debugging

- Enable DEBUG mode in the .env file
- Check logs for error messages and stack traces
- Use the test scripts in the scripts directory to isolate issues

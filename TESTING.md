# Testing Guide for Genesilico OCR

This document explains how to run the tests for the Genesilico OCR project.

## Prerequisites

Before running the tests, make sure you have the following dependencies installed:

```bash
pip install pytest pytest-asyncio pytest-cov
```

## Running the Tests

### Run All Tests

To run all tests:

```bash
python -m pytest
```

### Run Specific Test Files

To run tests from a specific file:

```bash
python -m pytest tests/test_mongo_helpers.py
python -m pytest tests/test_ocr_service.py
python -m pytest tests/test_document_processor.py
python -m pytest tests/test_utils.py
```

### Run with Coverage Report

To check test coverage:

```bash
python -m pytest --cov=app tests/
```

For a more detailed coverage report:

```bash
python -m pytest --cov=app --cov-report=term-missing tests/
```

To generate an HTML coverage report:

```bash
python -m pytest --cov=app --cov-report=html tests/
```

This will create an `htmlcov` directory with an interactive HTML report.

## Test Structure

The test suite is organized into the following files:

1. `test_mongo_helpers.py` - Tests for MongoDB helper functions
2. `test_ocr_service.py` - Tests for the OCR service
3. `test_field_extractor.py` - Tests for the AI field extractor
4. `test_document_processor.py` - Tests for the document processor
5. `test_utils.py` - Tests for utility functions
6. `test_ocr_processing.py` - Tests for OCR processing

Some test files are set to skip certain tests that require external dependencies like FastAPI. To run these tests, you'll need to remove the `pytestmark = pytest.mark.skip(...)` line from the relevant files and install the required dependencies.

## Dependencies

The tests use mocks to avoid requiring actual connections to:

1. MongoDB
2. Google Generative AI (Gemini)
3. OpenAI API
4. PDF conversion tools

This allows you to run the tests without setting up these external services.

## Adding New Tests

When adding new tests:

1. Follow the existing patterns of using fixtures and mocks
2. Use `@pytest.mark.asyncio` for asynchronous tests
3. Make sure to test both success cases and error handling

For more complex tests that interact with the FastAPI application, you may need to modify the test files to remove the `skip` markers and properly set up the FastAPI test client.
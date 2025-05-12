"""Global test configuration and fixtures."""

import os
import sys
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

# Create pytest.ini content for asyncio
if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), "pytest.ini")):
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "pytest.ini"), "w") as f:
        f.write("""[pytest]
markers =
    asyncio: mark a test as an asyncio test
""")

# Mock the HarmCategory and HarmBlockThreshold classes
class HarmCategoryMock:
    HARM_CATEGORY_HARASSMENT = "HARM_CATEGORY_HARASSMENT"
    HARM_CATEGORY_HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
    HARM_CATEGORY_SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
    HARM_CATEGORY_DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"

class HarmBlockThresholdMock:
    BLOCK_NONE = "BLOCK_NONE"
    BLOCK_LOW = "BLOCK_LOW"
    BLOCK_MEDIUM = "BLOCK_MEDIUM"
    BLOCK_HIGH = "BLOCK_HIGH"

# Mock the GenAI types module
class TypesMock:
    def __init__(self):
        self.HarmCategory = HarmCategoryMock
        self.HarmBlockThreshold = HarmBlockThresholdMock

# Mock pdf2image module
sys.modules['pdf2image'] = MagicMock()
pdf2image_mock = sys.modules['pdf2image']
pdf2image_mock.convert_from_path = MagicMock(return_value=[MagicMock()])

# Mock google.generativeai
class GenAIMock:
    def __init__(self):
        self.types = TypesMock()
        
    def configure(self, **kwargs):
        pass
    
    class GenerativeModel:
        def __init__(self, *args, **kwargs):
            pass
        
        def generate_content(self, *args, **kwargs):
            return MagicMock(text="Mock OCR text for testing purposes.")

# Create the mock structure
genai_mock = GenAIMock()

# Create the mock for google module
if 'google' not in sys.modules:
    sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = genai_mock
sys.modules['google.api_core'] = MagicMock()
sys.modules['google.api_core.exceptions'] = MagicMock()
sys.modules['google.api_core.exceptions'].InvalidArgument = Exception

# Create the mock for HarmCategory and HarmBlockThreshold globals
globals()['HarmCategory'] = HarmCategoryMock
globals()['HarmBlockThreshold'] = HarmBlockThresholdMock

# Mock settings module with API keys
class SettingsMock:
    def __init__(self):
        self.GEMINI_API_KEY = "mock_gemini_api_key"
        self.OPENAI_API_KEY = "mock_openai_api_key"
        self.ENV = "test"
        self.LOG_LEVEL = "INFO"
        self.UPLOAD_DIR = "/tmp/uploads"

# Patch app.config.settings
@pytest.fixture(autouse=True)
def mock_settings():
    with patch("app.core.field_extractor.settings", new=SettingsMock()), \
         patch("app.core.ocr_service.settings", new=SettingsMock()), \
         patch("app.utils.file_utils.settings", new=SettingsMock()), \
         patch("app.utils.log_utils.settings", new=SettingsMock()), \
         patch("app.config.settings", new=SettingsMock()):
        yield SettingsMock()

# Patch HarmCategory and HarmBlockThreshold in all modules
@pytest.fixture(autouse=True)
def mock_harm_categories():
    with patch("app.core.ocr_service.HarmCategory", HarmCategoryMock), \
         patch("app.core.ocr_service.HarmBlockThreshold", HarmBlockThresholdMock):
        yield

# Mock MongoDB connection
@pytest.fixture(autouse=True)
def mock_db_connection():
    """Mock MongoDB connection."""
    with patch("app.core.database.connect_to_mongodb") as mock_connect:
        mock_connect.return_value = True
        yield mock_connect

# Create temporary directories for file operations
@pytest.fixture(autouse=True)
def setup_temp_dirs():
    """Set up temporary directories for file operations."""
    os.makedirs("/tmp/uploads", exist_ok=True)
    for path in ["/tmp/test.pdf", "/tmp/test.jpg", "/tmp/test1.pdf", "/tmp/test2.pdf"]:
        with open(path, "w") as f:
            f.write("dummy content")
    yield
    # We'll skip cleanup for simplicity in tests

# Mock API routes structure
@pytest.fixture(autouse=True)
def mock_api_routes():
    """Mock API routes structure to match imports in tests."""
    # Create the module structure
    if 'app.api.routes' not in sys.modules:
        sys.modules['app.api.routes'] = MagicMock()
    
    sys.modules['app.api.routes.documents'] = MagicMock()
    sys.modules['app.api.routes.ocr'] = MagicMock()
    
    yield

# Mock normalization function
@pytest.fixture(autouse=True)
def mock_normalization():
    """Mock normalization utilities."""
    with patch("app.core.document_processor.normalize_array_fields") as mock_normalize:
        # Configure normalization mock to return the input unchanged
        mock_normalize.side_effect = lambda x: x
        
        # Import normalization in document_processor
        import app.core.document_processor
        app.core.document_processor.normalize_array_fields = mock_normalize
        
        yield mock_normalize

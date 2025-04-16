import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Project root directory
ROOT_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "GenesilicoCRAgent"
    ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Database settings
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "genesilico_ocr"

    # Mistral AI settings
    MISTRAL_API_KEY: str
    OPENAI_API_KEY: str

    # File storage settings
    UPLOAD_DIR: Path = ROOT_DIR / "data" / "documents"
    OCR_RESULTS_DIR: Path = ROOT_DIR / "data" / "ocr_results"
    TRF_OUTPUTS_DIR: Path = ROOT_DIR / "data" / "trf_outputs"
    MAX_UPLOAD_SIZE_MB: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OCR_RESULTS_DIR, exist_ok=True)
os.makedirs(settings.TRF_OUTPUTS_DIR, exist_ok=True)

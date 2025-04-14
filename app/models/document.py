from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid


class Document(BaseModel):
    """Document model representing an uploaded document."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str
    file_path: str
    file_size: int
    file_type: str
    upload_time: datetime = Field(default_factory=datetime.now)
    status: str = "uploaded"  # uploaded, processing, processed, failed
    ocr_result_id: Optional[str] = None
    trf_data_id: Optional[str] = None

    class Config:
        populate_by_name = True


class OCRResult(BaseModel):
    """OCR result model representing the result of OCR processing."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    text: str
    confidence: float
    processing_time: float
    created_at: datetime = Field(default_factory=datetime.now)
    pages: List[dict] = []  # List of pages with text and positions

    class Config:
        populate_by_name = True


class ProcessingStatus(BaseModel):
    """Processing status model for tracking document processing."""
    document_id: str
    status: str  # uploaded, ocr_processing, ocr_completed, extraction_processing, extraction_completed, validation_processing, validation_completed, agent_processing, agent_completed, failed
    message: Optional[str] = None
    progress: float = 0.0
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        populate_by_name = True

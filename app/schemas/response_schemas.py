from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class StatusEnum(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ProcessingStatusEnum(str, Enum):
    UPLOADED = "uploaded"
    OCR_PROCESSING = "ocr_processing"
    OCR_COMPLETED = "ocr_completed"
    EXTRACTION_PROCESSING = "extraction_processing"
    EXTRACTION_COMPLETED = "extraction_completed"
    VALIDATION_PROCESSING = "validation_processing"
    VALIDATION_COMPLETED = "validation_completed"
    AGENT_PROCESSING = "agent_processing"
    AGENT_COMPLETED = "agent_completed"
    COMPLETED = "completed"
    FAILED = "failed"


class BaseResponse(BaseModel):
    """Base response schema for all API responses."""
    status: StatusEnum
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class DocumentUploadResponse(BaseResponse):
    """Response schema for document upload."""
    document_id: str
    file_name: str
    file_size: int
    file_type: str


class MultipleDocumentUploadResponse(BaseResponse):
    """Response schema for multiple document upload as a group."""
    group_id: str
    group_name: str
    documents: List[Dict[str, Any]]
    total_files: int
    total_size: int


class ProcessingStatusResponse(BaseResponse):
    """Response schema for processing status."""
    document_id: str
    status_value: ProcessingStatusEnum  # Renamed from 'status' to avoid conflict with BaseResponse.status
    progress: float
    details: Optional[Dict[str, Any]] = None


class OCRResultResponse(BaseResponse):
    """Response schema for OCR result."""
    document_id: str
    ocr_result_id: str
    text_sample: str  # First 500 characters
    confidence: float
    processing_time: float
    page_count: int


class FieldExtractionResponse(BaseResponse):
    """Response schema for field extraction result."""
    document_id: str
    trf_data_id: str
    extracted_fields: Dict[str, float]  # field_name: confidence
    missing_required_fields: List[str]
    low_confidence_fields: List[str]
    extraction_time: float


class AgentQueryResponse(BaseResponse):
    """Response schema for AI agent query."""
    document_id: str
    query: str
    response: str
    suggested_actions: Optional[List[Dict[str, Any]]] = None
    field_updates: Optional[Dict[str, Any]] = None


class TRFDataResponse(BaseResponse):
    """Response schema for TRF data."""
    document_id: str
    trf_data_id: str
    trf_data: Dict[str, Any]
    missing_required_fields: List[str]
    low_confidence_fields: List[str]
    extraction_confidence: float
    completion_percentage: float


class FieldUpdateResponse(BaseResponse):
    """Response schema for field update."""
    document_id: str
    field_path: str
    previous_value: Any
    new_value: Any
    confidence: float

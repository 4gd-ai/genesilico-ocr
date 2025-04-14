from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class DocumentType(str, Enum):
    PDF = "pdf"
    JPG = "jpg"
    JPEG = "jpeg"


class DocumentUploadRequest(BaseModel):
    """Request schema for document upload."""
    # Note: The actual file will be handled by FastAPI's File parameter
    document_type: Optional[DocumentType] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ProcessDocumentRequest(BaseModel):
    """Request schema for processing a document."""
    document_id: str
    force_reprocess: Optional[bool] = False


class AgentQueryRequest(BaseModel):
    """Request schema for querying the AI agent."""
    document_id: str
    query: str
    context: Optional[Dict[str, Any]] = None


class FieldUpdateRequest(BaseModel):
    """Request schema for updating a field in the TRF data."""
    document_id: str
    field_path: str  # e.g., "patientInformation.patientName.firstName"
    field_value: Any
    confidence: Optional[float] = None
    notes: Optional[str] = None

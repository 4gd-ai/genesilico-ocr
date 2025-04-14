"""Utility functions for logging."""

import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional

from ..config import settings

# Configure root logger
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"), exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL)
    
    # Add file handler if in production
    if settings.ENV == "production":
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        
        # Create file handler
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        file_handler.setLevel(settings.LOG_LEVEL)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
    
    return logger


def log_request(logger: logging.Logger, request_id: str, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an API request.
    
    Args:
        logger: Logger instance
        request_id: Unique request ID
        method: HTTP method
        path: Request path
        params: Request parameters
    """
    log_data = {
        "request_id": request_id,
        "method": method,
        "path": path,
        "timestamp": datetime.now().isoformat(),
    }
    
    if params:
        log_data["params"] = params
    
    logger.info(f"API Request: {json.dumps(log_data)}")


def log_response(logger: logging.Logger, request_id: str, status_code: int, response_time: float, response_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an API response.
    
    Args:
        logger: Logger instance
        request_id: Unique request ID
        status_code: HTTP status code
        response_time: Response time in seconds
        response_data: Response data
    """
    log_data = {
        "request_id": request_id,
        "status_code": status_code,
        "response_time": response_time,
        "timestamp": datetime.now().isoformat(),
    }
    
    if response_data:
        # Truncate large response data
        if isinstance(response_data, dict) and "text" in response_data and len(response_data["text"]) > 500:
            response_data["text"] = response_data["text"][:500] + "..."
        log_data["response"] = response_data
    
    logger.info(f"API Response: {json.dumps(log_data)}")


def log_document_processing(logger: logging.Logger, document_id: str, stage: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
    """
    Log document processing events.
    
    Args:
        logger: Logger instance
        document_id: Document ID
        stage: Processing stage
        status: Processing status
        details: Processing details
    """
    log_data = {
        "document_id": document_id,
        "stage": stage,
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }
    
    if details:
        log_data["details"] = details
    
    logger.info(f"Document Processing: {json.dumps(log_data)}")


def log_ocr_result(logger: logging.Logger, document_id: str, ocr_result_id: str, confidence: float, processing_time: float) -> None:
    """
    Log OCR result events.
    
    Args:
        logger: Logger instance
        document_id: Document ID
        ocr_result_id: OCR result ID
        confidence: OCR confidence
        processing_time: Processing time in seconds
    """
    log_data = {
        "document_id": document_id,
        "ocr_result_id": ocr_result_id,
        "confidence": confidence,
        "processing_time": processing_time,
        "timestamp": datetime.now().isoformat(),
    }
    
    logger.info(f"OCR Result: {json.dumps(log_data)}")


def log_field_extraction(logger: logging.Logger, document_id: str, trf_data_id: str, extraction_confidence: float, missing_fields: int, low_confidence_fields: int) -> None:
    """
    Log field extraction events.
    
    Args:
        logger: Logger instance
        document_id: Document ID
        trf_data_id: TRF data ID
        extraction_confidence: Extraction confidence
        missing_fields: Number of missing fields
        low_confidence_fields: Number of low-confidence fields
    """
    log_data = {
        "document_id": document_id,
        "trf_data_id": trf_data_id,
        "extraction_confidence": extraction_confidence,
        "missing_fields": missing_fields,
        "low_confidence_fields": low_confidence_fields,
        "timestamp": datetime.now().isoformat(),
    }
    
    logger.info(f"Field Extraction: {json.dumps(log_data)}")


# Create and export application loggers
app_logger = get_logger("app")
api_logger = get_logger("api")
ocr_logger = get_logger("ocr")
extraction_logger = get_logger("extraction")
agent_logger = get_logger("agent")

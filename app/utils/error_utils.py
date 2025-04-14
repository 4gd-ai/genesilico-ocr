"""Utility functions for error handling."""

import traceback
import logging
from typing import Dict, Any, Optional, Type, Union
from fastapi import HTTPException

from ..config import settings

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)


def format_exception(e: Exception) -> str:
    """
    Format an exception with traceback for logging.
    
    Args:
        e: Exception to format
        
    Returns:
        Formatted exception message with traceback
    """
    return f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


def log_exception(e: Exception, context: Optional[str] = None) -> None:
    """
    Log an exception with context information.
    
    Args:
        e: Exception to log
        context: Context information
    """
    error_message = format_exception(e)
    if context:
        error_message = f"{context}: {error_message}"
    
    logger.error(error_message)


def handle_exception(e: Exception, context: Optional[str] = None, raise_http: bool = True) -> Dict[str, Any]:
    """
    Handle an exception by logging it and optionally raising an HTTP exception.
    
    Args:
        e: Exception to handle
        context: Context information
        raise_http: Whether to raise an HTTP exception
        
    Returns:
        Error response data
        
    Raises:
        HTTPException: If raise_http is True
    """
    # Log the exception
    log_exception(e, context)
    
    # Create error response
    error_response = {
        "error": str(e),
        "error_type": type(e).__name__
    }
    
    # Add context if available
    if context:
        error_response["context"] = context
    
    # Raise HTTP exception if requested
    if raise_http:
        status_code = 500
        if isinstance(e, HTTPException):
            status_code = e.status_code
        
        raise HTTPException(status_code=status_code, detail=str(e))
    
    return error_response


def api_error_response(error_message: str, status_code: int = 400, error_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a standardized API error response.
    
    Args:
        error_message: Error message
        status_code: HTTP status code
        error_type: Type of error
        
    Returns:
        API error response data
    """
    return {
        "status": "error",
        "message": error_message,
        "status_code": status_code,
        "error_type": error_type or "APIError"
    }


class APIError(Exception):
    """Base class for API errors."""
    
    def __init__(self, message: str, status_code: int = 400, error_type: Optional[str] = None):
        """Initialize API error."""
        self.message = message
        self.status_code = status_code
        self.error_type = error_type or self.__class__.__name__
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "status": "error",
            "message": self.message,
            "status_code": self.status_code,
            "error_type": self.error_type
        }

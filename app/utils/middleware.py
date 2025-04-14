"""Middleware for the application."""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .log_utils import api_logger, log_request, log_response


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""
    
    def __init__(self, app: ASGIApp):
        """Initialize request logger middleware."""
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and log request/response information.
        
        Args:
            request: HTTP request
            call_next: Next middleware or handler
            
        Returns:
            HTTP response
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Start timer
        start_time = time.time()
        
        # Log request
        log_request(
            api_logger,
            request_id,
            request.method,
            request.url.path,
            dict(request.query_params)
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            # Log response
            log_response(
                api_logger,
                request_id,
                response.status_code,
                response_time
            )
            
            return response
            
        except Exception as e:
            # Calculate response time
            response_time = time.time() - start_time
            
            # Log error response
            log_response(
                api_logger,
                request_id,
                500,
                response_time,
                {"error": str(e)}
            )
            
            # Re-raise exception
            raise

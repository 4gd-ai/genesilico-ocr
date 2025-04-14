"""Main FastAPI application module."""

import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .config import settings
from .api.routes import api_router
from .core.database import connect_to_mongodb, close_mongodb_connection
from .utils.middleware import RequestLoggerMiddleware
from .utils.error_utils import format_exception
from .utils.log_utils import app_logger


# Create FastAPI app
app = FastAPI(
    title="Genesilico OCR + AI Agent Service",
    description="API for OCR processing and AI-assisted field extraction from Test Requisition Forms",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify the allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logger middleware
app.add_middleware(RequestLoggerMiddleware)

# Include API router
app.include_router(api_router)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    # Log the exception
    app_logger.error(f"Unhandled exception: {format_exception(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    app_logger.info("Starting Genesilico OCR + AI Agent Service...")
    
    # Connect to MongoDB
    mongodb_connected = await connect_to_mongodb()
    if not mongodb_connected:
        app_logger.error("Failed to connect to MongoDB")
        raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")
    
    # Create data directories if they don't exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OCR_RESULTS_DIR, exist_ok=True)
    os.makedirs(settings.TRF_OUTPUTS_DIR, exist_ok=True)
    
    app_logger.info(f"Genesilico OCR + AI Agent Service started on {settings.API_HOST}:{settings.API_PORT}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    app_logger.info("Shutting down Genesilico OCR + AI Agent Service...")
    
    # Close MongoDB connection
    await close_mongodb_connection()
    
    app_logger.info("Genesilico OCR + AI Agent Service shut down")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Genesilico OCR + AI Agent Service",
        "version": "0.1.0",
        "status": "running"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.ENV == "development"
    )

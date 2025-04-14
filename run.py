#!/usr/bin/env python
"""Script to run the Genesilico OCR + AI Agent Service."""

import os
import argparse
import uvicorn

from app.config import settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Genesilico OCR + AI Agent Service")
    
    parser.add_argument(
        "--host",
        type=str,
        default=settings.API_HOST,
        help=f"Host to bind to (default: {settings.API_HOST})"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=settings.API_PORT,
        help=f"Port to bind to (default: {settings.API_PORT})"
    )
    
    parser.add_argument(
        "--env",
        type=str,
        choices=["development", "production"],
        default=settings.ENV,
        help=f"Environment to run in (default: {settings.ENV})"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)"
    )
    
    return parser.parse_args()


def main():
    """Run the application."""
    args = parse_args()
    
    # Determine whether to enable auto-reload
    reload = args.reload or args.env == "development"
    
    # Print startup information
    print(f"Starting Genesilico OCR + AI Agent Service...")
    print(f"Environment: {args.env}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Auto-reload: {'enabled' if reload else 'disabled'}")
    print()
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=reload
    )


if __name__ == "__main__":
    main()

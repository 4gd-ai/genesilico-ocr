"""Utility functions for file handling."""

import os
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from ..config import settings


def generate_unique_filename(original_filename: str) -> str:
    """
    Generate a unique filename based on the original filename.
    
    Args:
        original_filename: Original filename
        
    Returns:
        Unique filename
    """
    # Get file extension
    file_extension = original_filename.split(".")[-1].lower()
    
    # Generate unique ID
    unique_id = str(uuid.uuid4())
    
    # Create unique filename
    unique_filename = f"{unique_id}.{file_extension}"
    
    return unique_filename


def save_uploaded_file(file_data: bytes, filename: str, directory: Optional[Path] = None) -> Tuple[str, int]:
    """
    Save an uploaded file to disk.
    
    Args:
        file_data: File data
        filename: Filename
        directory: Directory to save the file (defaults to UPLOAD_DIR)
        
    Returns:
        Tuple of (file_path, file_size)
    """
    # Use default directory if not specified
    if directory is None:
        directory = settings.UPLOAD_DIR
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Generate unique filename
    unique_filename = generate_unique_filename(filename)
    
    # Create file path
    file_path = Path(directory) / unique_filename
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(file_data)
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    return str(file_path), file_size


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get file information
    file_info = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_size": os.path.getsize(file_path),
        "file_extension": os.path.splitext(file_path)[1].lower()[1:],
        "created_time": os.path.getctime(file_path),
        "modified_time": os.path.getmtime(file_path)
    }
    
    return file_info


def delete_file(file_path: str) -> bool:
    """
    Delete a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file was deleted, False otherwise
    """
    try:
        # Check if file exists
        if not os.path.isfile(file_path):
            return False
        
        # Delete file
        os.remove(file_path)
        
        return True
    except Exception:
        return False

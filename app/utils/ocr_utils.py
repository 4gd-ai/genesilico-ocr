"""Utility functions for OCR processing."""

import os
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pdf2image
from PIL import Image


def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[str]:
    """
    Convert a PDF file to a list of image files.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: DPI for conversion (higher is better quality but larger files)
        
    Returns:
        List of image file paths
    """
    # Create temporary directory for images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert PDF to images
        images = pdf2image.convert_from_path(
            pdf_path,
            dpi=dpi,
            output_folder=temp_dir,
            fmt="png"
        )
        
        # Get image file paths
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(temp_dir, f"page_{i+1}.png")
            image.save(image_path, "PNG")
            image_paths.append(image_path)
        
        return image_paths


def optimize_image_for_ocr(image_path: str, output_path: Optional[str] = None) -> str:
    """
    Optimize an image for OCR processing.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the optimized image (if None, overwrites the input)
        
    Returns:
        Path to the optimized image
    """
    # Use input path as output path if not specified
    if output_path is None:
        output_path = image_path
    
    # Open the image
    image = Image.open(image_path)
    
    # Convert to RGB if not already (some OCR engines prefer RGB)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Apply basic enhancement for OCR
    # For a production system, this would be more sophisticated
    # and might include deskewing, noise removal, etc.
    
    # Save the optimized image
    image.save(output_path, "PNG")
    
    return output_path


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Get the dimensions of an image.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Tuple of (width, height)
    """
    # Open the image
    image = Image.open(image_path)
    
    # Get dimensions
    width, height = image.size
    
    return width, height

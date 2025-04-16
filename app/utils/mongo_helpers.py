"""Helper functions for MongoDB operations."""

from bson import ObjectId
from typing import Any, Dict, List, Union


def json_serialize_mongodb_object(obj: Any) -> Any:
    """
    Recursively convert MongoDB documents to JSON-serializable types.
    
    This function converts:
    - ObjectId to string
    - datetime to ISO format string
    - Recursively processes lists and dictionaries
    
    Args:
        obj: Any object that might contain MongoDB types
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: json_serialize_mongodb_object(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize_mongodb_object(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # Handle datetime objects
        return obj.isoformat()
    else:
        return obj


def sanitize_mongodb_document(doc: Dict) -> Dict:
    """
    Convert a MongoDB document to a JSON-serializable dictionary.
    
    This function:
    1. Converts MongoDB ObjectId to string
    2. Handles nested documents and arrays
    3. Converts other MongoDB specific types as needed
    
    Args:
        doc: MongoDB document
        
    Returns:
        JSON-serializable dictionary
    """
    if doc is None:
        return {}
    
    return json_serialize_mongodb_object(doc)

"""Tests for MongoDB helper functions."""

import pytest
from datetime import datetime
import json
from bson import ObjectId
from unittest.mock import patch

from app.utils.mongo_helpers import sanitize_mongodb_document


class TestMongoHelpers:
    def sanitize_mongodb_document(data):
        if data is None:
            return None
        elif isinstance(data, ObjectId):
            return str(data)
        elif isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, list):
            return [sanitize_mongodb_document(item) for item in data]
        elif isinstance(data, dict):
            return {key: sanitize_mongodb_document(value) for key, value in data.items()}
        else:
            return data
    
    def test_sanitize_mongodb_document_primitive(self):
        """Test sanitizing primitive values."""
        # Sanitize primitive values
        assert sanitize_mongodb_document(123) == 123
        assert sanitize_mongodb_document("test") == "test"
        assert sanitize_mongodb_document(True) is True
        assert sanitize_mongodb_document(3.14) == 3.14
    
    def test_sanitize_mongodb_document_object_id(self):
        """Test sanitizing ObjectId."""
        # Create ObjectId
        obj_id = ObjectId("507f1f77bcf86cd799439011")
        
        # Sanitize ObjectId
        result = sanitize_mongodb_document(obj_id)
        
        # Verify the result
        assert result == "507f1f77bcf86cd799439011"
        assert isinstance(result, str)
    
    def test_sanitize_mongodb_document_datetime(self):
        """Test sanitizing datetime."""
        # Create datetime
        dt = datetime(2023, 1, 1, 12, 0, 0)
        
        # Sanitize datetime
        result = sanitize_mongodb_document(dt)
        
        # Verify the result
        assert result == dt.isoformat()
        assert isinstance(result, str)
    
    def test_sanitize_mongodb_document_list(self):
        """Test sanitizing list."""
        # Create list with various types
        data = [
            123,
            "test",
            ObjectId("507f1f77bcf86cd799439011"),
            datetime(2023, 1, 1, 12, 0, 0),
            [1, 2, 3]
        ]
        
        # Sanitize list
        result = sanitize_mongodb_document(data)
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 5
        assert result[0] == 123
        assert result[1] == "test"
        assert result[2] == "507f1f77bcf86cd799439011"
        assert result[3] == datetime(2023, 1, 1, 12, 0, 0).isoformat()
        assert result[4] == [1, 2, 3]
    
    def test_sanitize_mongodb_document_dict(self):
        """Test sanitizing dictionary."""
        # Create dictionary with various types
        data = {
            "id": ObjectId("507f1f77bcf86cd799439011"),
            "name": "Test",
            "created_at": datetime(2023, 1, 1, 12, 0, 0),
            "tags": ["tag1", "tag2"],
            "nested": {
                "id": ObjectId("507f1f77bcf86cd799439022"),
                "value": 123
            }
        }
        
        # Sanitize dictionary
        result = sanitize_mongodb_document(data)
        
        # Verify the result
        assert isinstance(result, dict)
        assert result["id"] == "507f1f77bcf86cd799439011"
        assert result["name"] == "Test"
        assert result["created_at"] == datetime(2023, 1, 1, 12, 0, 0).isoformat()
        assert result["tags"] == ["tag1", "tag2"]
        assert isinstance(result["nested"], dict)
        assert result["nested"]["id"] == "507f1f77bcf86cd799439022"
        assert result["nested"]["value"] == 123
    
    def test_sanitize_mongodb_document_empty_dict(self):
        """Test sanitizing an empty dictionary."""
        # Sanitize an empty dictionary
        result = sanitize_mongodb_document({})
        
        # Verify the result is an empty dictionary, not None
        assert result == {}
        assert result is not None
    
    def test_sanitize_mongodb_document_json_serializable(self):
        """Test that sanitized document is JSON serializable."""
        # Create complex document
        data = {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "name": "Test",
            "created_at": datetime(2023, 1, 1, 12, 0, 0),
            "tags": ["tag1", "tag2"],
            "nested": {
                "id": ObjectId("507f1f77bcf86cd799439022"),
                "value": 123
            }
        }
        
        # Sanitize document
        result = sanitize_mongodb_document(data)
        
        # Verify JSON serialization works
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        
        # Verify deserialization works
        parsed = json.loads(json_str)
        assert parsed["_id"] == "507f1f77bcf86cd799439011"
        assert parsed["name"] == "Test"
        assert parsed["created_at"] == datetime(2023, 1, 1, 12, 0, 0).isoformat()

import os
import time
import uuid
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from ..config import settings
from ..core.database import documents_collection, ocr_results_collection, trf_data_collection
from ..models.document import Document, OCRResult, ProcessingStatus
from ..models.trf import PatientReport
from ..core.ocr_service import ocr_service
from ..core.field_extractor import FieldExtractor
from ..schemas.trf_schema import validate_trf_data

class DocumentProcessor:
    """Process documents through the OCR and field extraction pipeline."""
    
    @staticmethod
    async def process_document(document_id: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process a document through the OCR and field extraction pipeline.
        
        Args:
            document_id: ID of the document to process
            force_reprocess: Whether to force reprocessing if already processed
            
        Returns:
            Processing status
        """
        try:
            # Retrieve document from database
            document_data = await documents_collection.find_one({"id": document_id})
            if not document_data:
                return {"error": f"Document with ID {document_id} not found"}
            
            document = Document(**document_data)
            
            # Check if already processed and not forcing reprocess
            if document.status == "processed" and not force_reprocess:
                return {"message": f"Document {document_id} already processed", "status": "completed"}
            
            # Update document status
            await documents_collection.update_one(
                {"id": document_id},
                {"$set": {"status": "processing"}}
            )
            
            # Create processing status
            processing_status = ProcessingStatus(
                document_id=document_id,
                status="ocr_processing",
                message="Starting OCR processing",
                progress=0.1
            )
            
            # Start OCR processing
            start_time = time.time()
            
            ocr_result = await ocr_service.process_document(
                document.file_path, 
                document.file_type
            )
            
            ocr_result.document_id = document_id
            ocr_result.processing_time = time.time() - start_time
            
            # Save OCR result to database
            ocr_result_dict = ocr_result.dict()
            await ocr_results_collection.insert_one(ocr_result_dict)
            
            # Update document with OCR result ID
            await documents_collection.update_one(
                {"id": document_id},
                {"$set": {
                    "ocr_result_id": ocr_result.id,
                    "status": "ocr_processed"
                }}
            )
            
            # Update processing status
            processing_status.status = "extraction_processing"
            processing_status.message = "Starting field extraction"
            processing_status.progress = 0.4
            
            # Start field extraction
            field_extractor = FieldExtractor(ocr_result)
            start_extraction_time = time.time()
            
            # Extract basic fields using patterns
            trf_data, confidence_scores, extraction_stats = await field_extractor.extract_fields()
            
            # Extract detailed fields using specialized methods
            trf_data = await field_extractor.extract_patient_info(trf_data)
            trf_data = await field_extractor.extract_clinical_summary(trf_data)
            trf_data = await field_extractor.extract_physician_info(trf_data)
            trf_data = await field_extractor.extract_sample_info(trf_data)
            trf_data = await field_extractor.extract_hospital_info(trf_data)
            
            extraction_time = time.time() - start_extraction_time
            
            # Create PatientReport object
            patient_report = PatientReport(**trf_data)
            patient_report.document_id = document_id
            patient_report.ocr_result_id = ocr_result.id
            patient_report.extraction_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0
            patient_report.extraction_time = extraction_time
            
            # Add extracted fields tracking
            patient_report.extracted_fields = confidence_scores
            
            # Validate TRF data
            is_valid, missing_required_fields, validation_errors = validate_trf_data(trf_data)
            patient_report.missing_required_fields = missing_required_fields
            
            # Identify low confidence fields
            low_confidence_fields = field_extractor.get_low_confidence_fields()
            patient_report.low_confidence_fields = low_confidence_fields
            
            # Save TRF data to database
            trf_data_dict = patient_report.dict()
            await trf_data_collection.insert_one(trf_data_dict)
            
            # Update document with TRF data ID
            await documents_collection.update_one(
                {"id": document_id},
                {"$set": {
                    "trf_data_id": patient_report.id,
                    "status": "processed"
                }}
            )
            
            # Update processing status
            processing_status.status = "completed"
            processing_status.message = "Document processing completed"
            processing_status.progress = 1.0
            
            # Return processing status
            return {
                "document_id": document_id,
                "status": "completed",
                "ocr_result_id": ocr_result.id,
                "trf_data_id": patient_report.id,
                "extraction_confidence": patient_report.extraction_confidence,
                "missing_required_fields": missing_required_fields,
                "low_confidence_fields": low_confidence_fields,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            # Update document status on error
            if document_id:
                await documents_collection.update_one(
                    {"id": document_id},
                    {"$set": {"status": "failed"}}
                )
            
            # Return error
            return {
                "document_id": document_id,
                "status": "failed",
                "error": str(e)
            }
    
    @staticmethod
    async def get_document_status(document_id: str) -> Dict[str, Any]:
        """
        Get the status of a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Document status
        """
        try:
            # Retrieve document from database
            document_data = await documents_collection.find_one({"id": document_id})
            if not document_data:
                return {"error": f"Document with ID {document_id} not found"}
            
            document = Document(**document_data)
            
            result = {
                "document_id": document_id,
                "status": document.status,
                "file_name": document.file_name,
                "file_type": document.file_type,
                "upload_time": document.upload_time,
            }
            
            # Include OCR result if available
            if document.ocr_result_id:
                ocr_result_data = await ocr_results_collection.find_one({"id": document.ocr_result_id})
                if ocr_result_data:
                    result["ocr_result_id"] = document.ocr_result_id
                    result["ocr_confidence"] = ocr_result_data.get("confidence", 0.0)
            
            # Include TRF data if available
            if document.trf_data_id:
                trf_data = await trf_data_collection.find_one({"id": document.trf_data_id})
                if trf_data:
                    result["trf_data_id"] = document.trf_data_id
                    result["extraction_confidence"] = trf_data.get("extraction_confidence", 0.0)
                    result["missing_required_fields"] = trf_data.get("missing_required_fields", [])
                    result["low_confidence_fields"] = trf_data.get("low_confidence_fields", [])
            
            return result
            
        except Exception as e:
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e)
            }
    
    @staticmethod
    async def get_trf_data(document_id: str) -> Dict[str, Any]:
        """
        Get the TRF data for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            TRF data
        """
        try:
            # Retrieve document from database
            document_data = await documents_collection.find_one({"id": document_id})
            if not document_data:
                return {"error": f"Document with ID {document_id} not found"}
            
            document = Document(**document_data)
            
            # Check if TRF data exists
            if not document.trf_data_id:
                return {"error": f"TRF data not found for document {document_id}"}
            
            # Retrieve TRF data
            trf_data = await trf_data_collection.find_one({"id": document.trf_data_id})
            if not trf_data:
                return {"error": f"TRF data with ID {document.trf_data_id} not found"}
            
            return {
                "document_id": document_id,
                "trf_data_id": document.trf_data_id,
                "trf_data": trf_data
            }
            
        except Exception as e:
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e)
            }
    
    @staticmethod
    async def update_trf_field(document_id: str, field_path: str, field_value: Any, confidence: float = None) -> Dict[str, Any]:
        """
        Update a field in the TRF data.
        
        Args:
            document_id: ID of the document
            field_path: Path to the field to update (e.g., "patientInformation.patientName.firstName")
            field_value: New value for the field
            confidence: Confidence score for the update
            
        Returns:
            Update status
        """
        try:
            # Retrieve document from database
            document_data = await documents_collection.find_one({"id": document_id})
            if not document_data:
                return {"error": f"Document with ID {document_id} not found"}
            
            document = Document(**document_data)
            
            # Check if TRF data exists
            if not document.trf_data_id:
                return {"error": f"TRF data not found for document {document_id}"}
            
            # Retrieve TRF data
            trf_data = await trf_data_collection.find_one({"id": document.trf_data_id})
            if not trf_data:
                return {"error": f"TRF data with ID {document.trf_data_id} not found"}
            
            # Get current value
            current_value = None
            parts = field_path.split('.')
            current = trf_data
            
            for i, part in enumerate(parts[:-1]):
                # Handle array indices
                if '[' in part and ']' in part:
                    array_name = part.split('[')[0]
                    index_str = part.split('[')[1].split(']')[0]
                    
                    if array_name not in current:
                        return {"error": f"Field path '{field_path}' not found in TRF data"}
                        
                    try:
                        index = int(index_str)
                        if not isinstance(current[array_name], list) or index >= len(current[array_name]):
                            return {"error": f"Field path '{field_path}' not found in TRF data"}
                        current = current[array_name][index]
                    except (ValueError, IndexError):
                        return {"error": f"Field path '{field_path}' not found in TRF data"}
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
            
            last_part = parts[-1]
            if last_part in current:
                current_value = current[last_part]
            
            # Update the field
            current[last_part] = field_value
            
            # Update confidence if provided
            if confidence is not None:
                if "extracted_fields" not in trf_data:
                    trf_data["extracted_fields"] = {}
                trf_data["extracted_fields"][field_path] = confidence
            
            # Update the TRF data in the database
            await trf_data_collection.update_one(
                {"id": document.trf_data_id},
                {"$set": {
                    **trf_data, 
                    "updated_at": time.time()
                }}
            )
            
            return {
                "document_id": document_id,
                "trf_data_id": document.trf_data_id,
                "field_path": field_path,
                "previous_value": current_value,
                "new_value": field_value,
                "confidence": confidence,
                "status": "updated"
            }
            
        except Exception as e:
            return {
                "document_id": document_id,
                "field_path": field_path,
                "status": "error",
                "error": str(e)
            }
    
    @staticmethod
    async def list_documents(limit: int = 10, skip: int = 0, status: str = None) -> Dict[str, Any]:
        """
        List documents with optional filtering.
        
        Args:
            limit: Maximum number of documents to return
            skip: Number of documents to skip
            status: Filter by document status
            
        Returns:
            List of documents
        """
        try:
            # Build query
            query = {}
            if status:
                query["status"] = status
            
            # Count total documents
            total_count = await documents_collection.count_documents(query)
            
            # Get documents
            cursor = documents_collection.find(query).skip(skip).limit(limit).sort("upload_time", -1)
            documents = await cursor.to_list(length=limit)
            
            return {
                "total": total_count,
                "limit": limit,
                "skip": skip,
                "documents": documents
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

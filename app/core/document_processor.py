import os
import time
import uuid
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from ..config import settings
from ..core.database import documents_collection, document_groups_collection, ocr_results_collection, trf_data_collection
from ..models.document import Document, DocumentGroup, OCRResult, ProcessingStatus
from ..models.trf import PatientReport
from ..core.ocr_service import ocr_service
# from ..core.field_extractor import FieldExtractor
from ..schemas.trf_schema import validate_trf_data
from ..core.field_extractor import AIFieldExtractor
from ..utils.mongo_helpers import sanitize_mongodb_document

class DocumentProcessor:
    """Process documents through the OCR and field extraction pipeline."""
    
    @staticmethod
    async def get_document_status(document_id: str) -> Dict[str, Any]:
        """
        Get the status of a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Document status information
        """
        try:
            # Retrieve document from database
            document_data = await documents_collection.find_one({"id": document_id})
            if not document_data:
                return {"error": f"Document with ID {document_id} not found"}
            
            document = Document(**document_data)
            
            # Prepare base response
            response = {
                "document_id": document_id,
                "status": document.status,
                "file_name": document.file_name,
                "file_type": document.file_type,
                "file_size": document.file_size
            }
            
            # Add OCR result information if available
            if document.ocr_result_id:
                response["ocr_result_id"] = document.ocr_result_id
                
                # Get OCR result details
                ocr_result_data = await ocr_results_collection.find_one({"id": document.ocr_result_id})
                if ocr_result_data:
                    # Sanitize OCR result data to make it JSON serializable
                    sanitized_ocr_data = sanitize_mongodb_document(ocr_result_data)
                    response["ocr_processing_time"] = sanitized_ocr_data.get("processing_time", 0)
                    response["ocr_confidence"] = sanitized_ocr_data.get("confidence", 0)
                    response["page_count"] = len(sanitized_ocr_data.get("pages", []))
            
            # Add TRF data information if available
            if document.trf_data_id:
                response["trf_data_id"] = document.trf_data_id
                
                # Get TRF data details
                trf_data = await trf_data_collection.find_one({"id": document.trf_data_id})
                if trf_data:
                    # Sanitize TRF data to make it JSON serializable
                    sanitized_trf_data = sanitize_mongodb_document(trf_data)
                    response["extraction_confidence"] = sanitized_trf_data.get("extraction_confidence", 0)
                    response["missing_required_fields"] = sanitized_trf_data.get("missing_required_fields", [])
                    response["low_confidence_fields"] = sanitized_trf_data.get("low_confidence_fields", [])
            
            return response
            
        except Exception as e:
            # Return error details
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e)
            }
    
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
            
            # Safely handle OCR results
            if not ocr_result:
                await documents_collection.update_one(
                    {"id": document_id},
                    {"$set": {"status": "failed"}}
                )
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "error": "OCR processing failed to return results"
                }
            
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
            field_extractor = AIFieldExtractor(ocr_result, model_name="gpt-4o")
            start_extraction_time = time.time()
            
            # Add safety checks before extraction
            if not hasattr(field_extractor, 'extract_fields'):
                await documents_collection.update_one(
                    {"id": document_id},
                    {"$set": {"status": "failed"}}
                )
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "error": "Field extractor initialization failed"
                }
            
            # Extract basic fields using patterns
            try:
                trf_data = await field_extractor.extract_fields()
            except Exception as e:
                await documents_collection.update_one(
                    {"id": document_id},
                    {"$set": {"status": "failed"}}
                )
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "error": f"Field extraction failed: {str(e)}"
                }
            print("\n--- Extracted Data ---")
            print(json.dumps(trf_data, indent=2))
            
            patient_report = trf_data[0]

            if len(trf_data) >= 3:
                stats = trf_data[2]
                # Add extraction confidence to patient data
                patient_report["extraction_confidence"] = stats.get("high_confidence_fields", 0) / stats.get("total_fields", 1)
                patient_report["missing_required_fields"] = []
                
                # Add low confidence fields from the second item
                if len(trf_data) >= 2:
                    confidence_scores = trf_data[1]
                    patient_report["low_confidence_fields"] = [
                        field for field, score in confidence_scores.items() 
                        if isinstance(score, (int, float)) and 0 < score < 0.7
                    ]
            
            patient_report = PatientReport(**patient_report)
            await trf_data_collection.insert_one(patient_report.dict())
            
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
                "missing_required_fields": patient_report.missing_required_fields,
                "low_confidence_fields": patient_report.low_confidence_fields,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            # Update document status on error
            if 'document_id' in locals():
                await documents_collection.update_one(
                    {"id": document_id},
                    {"$set": {"status": "failed"}}
                )
            
            # Return detailed error
            return {
                "document_id": document_id,
                "status": "failed",
                "error": str(e)
            }
            
    @staticmethod
    async def get_trf_data(id: str) -> Dict[str, Any]:
        """
        Get the TRF data for a document or document group.
        
        Args:
            id: ID of the document or document group
            
        Returns:
            TRF data information
        """
        try:
            # First, try to retrieve as a document
            document_data = await documents_collection.find_one({"id": id})
            
            if document_data:
                # This is a regular document
                document = Document(**document_data)
                
                # Check if TRF data exists
                if not document.trf_data_id:
                    return {"error": f"TRF data not found for document {id}"}
                
                # Get TRF data
                trf_data = await trf_data_collection.find_one({"id": document.trf_data_id})
                if not trf_data:
                    return {"error": f"TRF data with ID {document.trf_data_id} not found"}
                
                # Sanitize the MongoDB document to make it JSON serializable
                sanitized_trf_data = sanitize_mongodb_document(trf_data)
                
                # Return TRF data
                return {
                    "document_id": id,
                    "trf_data_id": document.trf_data_id,
                    "trf_data": sanitized_trf_data
                }
            
            # If not found as a document, try as a document group
            group_data = await document_groups_collection.find_one({"id": id})
            if group_data:
                # This is a document group
                group = DocumentGroup(**group_data)
                
                # Check if TRF data exists for the group
                if not group.trf_data_id:
                    return {"error": f"TRF data not found for document group {id}"}
                
                # Get TRF data
                trf_data = await trf_data_collection.find_one({"id": group.trf_data_id})
                if not trf_data:
                    return {"error": f"TRF data with ID {group.trf_data_id} not found"}
                
                # Sanitize the MongoDB document to make it JSON serializable
                sanitized_trf_data = sanitize_mongodb_document(trf_data)
                
                # Return TRF data
                return {
                    "group_id": id,
                    "document_id": id,  # For backward compatibility
                    "trf_data_id": group.trf_data_id,
                    "trf_data": sanitized_trf_data
                }
            
            # Not found as either a document or group
            return {"error": f"Document or document group with ID {id} not found"}
            
        except Exception as e:
            # Return error details
            return {
                "document_id": id,
                "status": "error",
                "error": str(e)
            }
            
    @staticmethod
    async def process_document_group(group_id: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process a group of documents through the OCR and field extraction pipeline.
        The documents are processed as a single entity.
        
        Args:
            group_id: ID of the document group to process
            force_reprocess: Whether to force reprocessing if already processed
            
        Returns:
            Processing status for the group
        """
        try:
            # Retrieve document group from database
            group_data = await document_groups_collection.find_one({"id": group_id})
            if not group_data:
                return {"error": f"Document group with ID {group_id} not found"}
            
            document_group = DocumentGroup(**group_data)
            
            # Check if already processed and not forcing reprocess
            if document_group.status == "processed" and not force_reprocess:
                return {"message": f"Document group {group_id} already processed", "status": "completed"}
            
            # Update document group status
            await document_groups_collection.update_one(
                {"id": group_id},
                {"$set": {"status": "processing"}}
            )
            
            # Get all documents in the group
            document_ids = document_group.document_ids
            if not document_ids:
                return {"error": f"No documents found in group {group_id}"}
            
            # Retrieve all documents
            documents = []
            for doc_id in document_ids:
                doc_data = await documents_collection.find_one({"id": doc_id})
                if doc_data:
                    documents.append(Document(**doc_data))
            
            if not documents:
                return {"error": f"No valid documents found in group {group_id}"}
            
            # Update status for all documents
            for document in documents:
                await documents_collection.update_one(
                    {"id": document.id},
                    {"$set": {"status": "processing"}}
                )
            
            # Start OCR processing for all documents
            start_time = time.time()
            
            # Process each document through OCR
            all_ocr_results = []
            combined_text = ""
            
            for document in documents:
                # Process each document
                ocr_result = await ocr_service.process_document(
                    document.file_path, 
                    document.file_type
                )
                
                # Safely handle OCR results
                if not ocr_result:
                    await documents_collection.update_one(
                        {"id": document.id},
                        {"$set": {"status": "failed"}}
                    )
                    continue
                
                ocr_result.document_id = document.id
                ocr_result.processing_time = time.time() - start_time
                
                # Save OCR result to database
                ocr_result_dict = ocr_result.dict()
                await ocr_results_collection.insert_one(ocr_result_dict)
                
                # Update document with OCR result ID
                await documents_collection.update_one(
                    {"id": document.id},
                    {"$set": {
                        "ocr_result_id": ocr_result.id,
                        "status": "ocr_processed"
                    }}
                )
                
                all_ocr_results.append(ocr_result)
                combined_text += ocr_result.text + "\n\n"
            
            if not all_ocr_results:
                await document_groups_collection.update_one(
                    {"id": group_id},
                    {"$set": {"status": "failed"}}
                )
                return {
                    "group_id": group_id,
                    "status": "failed",
                    "error": "OCR processing failed for all documents in the group"
                }
            
            # Create a combined OCR result for the group
            combined_ocr_result = OCRResult(
                document_id=group_id,
                text=combined_text,
                confidence=sum(r.confidence for r in all_ocr_results) / len(all_ocr_results),
                processing_time=time.time() - start_time,
                pages=[]
            )
            
            # Combine page data
            page_num = 1
            for ocr_result in all_ocr_results:
                for page in ocr_result.pages:
                    # Increment page number for the combined result
                    page["page_num"] = page_num
                    combined_ocr_result.pages.append(page)
                    page_num += 1
            
            # Save combined OCR result to database
            combined_ocr_dict = combined_ocr_result.dict()
            await ocr_results_collection.insert_one(combined_ocr_dict)
            
            # Update document group with OCR result ID
            await document_groups_collection.update_one(
                {"id": group_id},
                {"$set": {
                    "ocr_result_id": combined_ocr_result.id,
                    "status": "ocr_processed"
                }}
            )
            
            # Start field extraction with the combined OCR result
            field_extractor = AIFieldExtractor(combined_ocr_result, model_name="gpt-4o")
            
            # Extract fields
            try:
                trf_data = await field_extractor.extract_fields()
            except Exception as e:
                await document_groups_collection.update_one(
                    {"id": group_id},
                    {"$set": {"status": "failed"}}
                )
                return {
                    "group_id": group_id,
                    "status": "failed",
                    "error": f"Field extraction failed: {str(e)}"
                }
            
            print("\n--- Extracted Data (Group) ---")
            print(json.dumps(trf_data, indent=2))
            
            patient_report = trf_data[0]

            if len(trf_data) >= 3:
                stats = trf_data[2]
                # Add extraction confidence to patient data
                patient_report["extraction_confidence"] = stats.get("high_confidence_fields", 0) / stats.get("total_fields", 1)
                patient_report["missing_required_fields"] = []
                
                # Add low confidence fields from the second item
                if len(trf_data) >= 2:
                    confidence_scores = trf_data[1]
                    patient_report["low_confidence_fields"] = [
                        field for field, score in confidence_scores.items() 
                        if isinstance(score, (int, float)) and 0 < score < 0.7
                    ]
            
            patient_report = PatientReport(**patient_report)
            await trf_data_collection.insert_one(patient_report.dict())
            
            # Update document group with TRF data ID
            await document_groups_collection.update_one(
                {"id": group_id},
                {"$set": {
                    "trf_data_id": patient_report.id,
                    "status": "processed"
                }}
            )
            
            # Update all documents in the group to reference the group's TRF data
            for document in documents:
                await documents_collection.update_one(
                    {"id": document.id},
                    {"$set": {
                        "trf_data_id": patient_report.id,
                        "status": "processed"
                    }}
                )
            
            # Return processing status
            return {
                "group_id": group_id,
                "status": "completed",
                "document_count": len(documents),
                "ocr_result_id": combined_ocr_result.id,
                "trf_data_id": patient_report.id,
                "extraction_confidence": patient_report.extraction_confidence,
                "missing_required_fields": patient_report.missing_required_fields,
                "low_confidence_fields": patient_report.low_confidence_fields,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            # Update group status on error
            if 'group_id' in locals():
                await document_groups_collection.update_one(
                    {"id": group_id},
                    {"$set": {"status": "failed"}}
                )
            
            # Return detailed error
            return {
                "group_id": group_id,
                "status": "failed",
                "error": str(e)
            }
            
    @staticmethod
    async def list_documents(limit: int, skip: int, status: Optional[str] = None) -> Dict[str, Any]:
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
            # Prepare filter
            filter_dict = {}
            if status:
                filter_dict["status"] = status
            
            # Get total count
            total = await documents_collection.count_documents(filter_dict)
            
            # Get documents
            cursor = documents_collection.find(filter_dict).skip(skip).limit(limit)
            documents = []
            
            async for doc in cursor:
                # Sanitize each document to ensure it's JSON serializable
                sanitized_doc = sanitize_mongodb_document(doc)
                documents.append(sanitized_doc)
            
            # Return documents
            return {
                "total": total,
                "limit": limit,
                "skip": skip,
                "documents": documents
            }
            
        except Exception as e:
            # Return error details
            return {
                "error": str(e)
            }
            
    @staticmethod
    async def update_trf_field(document_id: str, field_path: str, field_value: str, confidence: Optional[float] = None) -> Dict[str, Any]:
        """
        Update a field in the TRF data.
        
        Args:
            document_id: ID of the document
            field_path: Path to the field to update
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
            
            # Get TRF data
            trf_data = await trf_data_collection.find_one({"id": document.trf_data_id})
            if not trf_data:
                return {"error": f"TRF data with ID {document.trf_data_id} not found"}
            
            # Parse field path
            path_parts = field_path.split(".")
            
            # Navigate to the field and update it
            current = trf_data
            previous_value = None
            
            # Navigate through nested dictionary
            for i, part in enumerate(path_parts):
                if i == len(path_parts) - 1:
                    # Found the field to update
                    previous_value = current.get(part, None)
                    current[part] = field_value
                else:
                    # Navigate deeper
                    if part not in current or not isinstance(current[part], dict):
                        current[part] = {}
                    current = current[part]
            
            # Update confidence if provided
            if confidence is not None:
                # Update field confidence
                extracted_fields = trf_data.get("extracted_fields", {})
                extracted_fields[field_path] = confidence
                trf_data["extracted_fields"] = extracted_fields
                
                # Update overall confidence
                confidences = list(extracted_fields.values())
                if confidences:
                    trf_data["extraction_confidence"] = sum(confidences) / len(confidences)
            
            # Remove field from missing/low confidence if applicable
            if "missing_required_fields" in trf_data and field_path in trf_data["missing_required_fields"]:
                trf_data["missing_required_fields"].remove(field_path)
            
            if "low_confidence_fields" in trf_data and field_path in trf_data["low_confidence_fields"]:
                trf_data["low_confidence_fields"].remove(field_path)
            
            # Update TRF data in database
            await trf_data_collection.update_one(
                {"id": document.trf_data_id},
                {"$set": trf_data}
            )
            
            # Return update status
            return {
                "document_id": document_id,
                "trf_data_id": document.trf_data_id,
                "field_path": field_path,
                "previous_value": sanitize_mongodb_document(previous_value),
                "new_value": field_value,
                "confidence": confidence
            }
            
        except Exception as e:
            # Return error details
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e)
            }
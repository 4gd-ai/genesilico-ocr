"""API routes for document upload and management."""

import os
import shutil
import time
import uuid
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio

from fastapi import APIRouter, File, UploadFile, Form, Query, Path as PathParam, HTTPException, BackgroundTasks, Body, Request
from fastapi.responses import JSONResponse

from ..config import settings
from ..core.database import documents_collection, document_groups_collection, ocr_results_collection, trf_data_collection, patientreports_collection
from ..core.document_processor import DocumentProcessor
from ..models.document import Document, DocumentGroup
from ..utils.mongo_helpers import sanitize_mongodb_document
from ..utils.normalization import normalize_array_fields
from ..schemas.request_schemas import DocumentUploadRequest, ProcessDocumentRequest
from ..schemas.response_schemas import (
    DocumentUploadResponse, MultipleDocumentUploadResponse, ProcessingStatusResponse, OCRResultResponse,
    FieldExtractionResponse, TRFDataResponse, StatusEnum, ProcessingStatusEnum
)
from ..models.trf import PatientReport
from ..utils.normalization import normalize_array_fields
from ..agent.suggestions import AgentSuggestions
from ..models.document import OCRResult
from ..schemas.trf_schema import set_field_value

router = APIRouter(prefix="/ocr/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    auto_process: bool = Form(False)
):
    """
    Upload a document for OCR processing.
    
    - **file**: The document file to upload (PDF, JPG, JPEG)
    - **description**: Optional description of the document
    - **tags**: Optional comma-separated tags
    - **auto_process**: Whether to automatically process the document after upload
    """
    try:
        # Validate file type
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["pdf", "jpg", "jpeg"]:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF, JPG, and JPEG are supported.")
        
        # Create unique ID for the document
        document_id = str(uuid.uuid4())
        
        # Create directory for document if it doesn't exist
        upload_dir = settings.UPLOAD_DIR
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the file
        file_path = Path(upload_dir) / f"{document_id}.{file_extension}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create document record
        document = Document(
            id=document_id,
            file_name=file.filename,
            file_path=str(file_path),
            file_size=os.path.getsize(file_path),
            file_type=file_extension,
            status="uploaded"
        )
        
        # Save document record to database
        await documents_collection.insert_one(document.dict())
        
        # Process document in background if auto_process is True
        if auto_process:
            background_tasks.add_task(DocumentProcessor.process_document, document_id)
        
        # Return response
        return DocumentUploadResponse(
            status=StatusEnum.SUCCESS,
            message="Document uploaded successfully",
            document_id=document_id,
            file_name=file.filename,
            file_size=document.file_size,
            file_type=file_extension
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")


@router.post("/upload-multi", response_model=MultipleDocumentUploadResponse)
async def upload_multiple_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    group_name: str = Form(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    auto_process: bool = Form(False)
):
    """
    Upload multiple documents as a single group for OCR processing.
    
    - **files**: List of document files to upload (JPG, JPEG)
    - **group_name**: Name for the document group
    - **description**: Optional description of the document group
    - **tags**: Optional comma-separated tags
    - **auto_process**: Whether to automatically process the document group after upload
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided for upload")
        
        # Create unique ID for the document group
        group_id = str(uuid.uuid4())
        
        # Create directory for the group if it doesn't exist
        upload_dir = Path(settings.UPLOAD_DIR) / group_id
        os.makedirs(upload_dir, exist_ok=True)
        
        document_ids = []
        document_details = []
        total_size = 0
        
        # Process each file
        for file in files:
            # Validate file type
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in ["jpg", "jpeg"]:
                raise HTTPException(status_code=400, detail=f"Unsupported file type for {file.filename}. Only JPG and JPEG are supported for multi-upload.")
            
            # Create unique ID for the document
            document_id = str(uuid.uuid4())
            document_ids.append(document_id)
            
            # Save the file
            file_path = upload_dir / f"{document_id}.{file_extension}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_size = os.path.getsize(file_path)
            total_size += file_size
            
            # Create document record
            document = Document(
                id=document_id,
                file_name=file.filename,
                file_path=str(file_path),
                file_size=file_size,
                file_type=file_extension,
                status="uploaded",
                group_id=group_id
            )
            
            # Save document record to database
            await documents_collection.insert_one(document.dict())
            
            # Add to document details
            document_details.append({
                "document_id": document_id,
                "file_name": file.filename,
                "file_size": file_size,
                "file_type": file_extension
            })
        
        # Create document group record
        document_group = DocumentGroup(
            id=group_id,
            name=group_name,
            description=description,
            document_ids=document_ids,
            status="created"
        )
        
        # Save document group record to database
        await document_groups_collection.insert_one(document_group.dict())
        
        # Process document group in background if auto_process is True
        if auto_process:
            background_tasks.add_task(DocumentProcessor.process_document_group, group_id)
        
        # Return response
        return MultipleDocumentUploadResponse(
            status=StatusEnum.SUCCESS,
            message=f"Successfully uploaded {len(files)} documents as group '{group_name}'",
            group_id=group_id,
            group_name=group_name,
            documents=document_details,
            total_files=len(files),
            total_size=total_size
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading multiple documents: {str(e)}")


@router.post("/process/{document_id}", response_model=ProcessingStatusResponse)
async def process_document(
    document_id: str,
    request: ProcessDocumentRequest,
    patient_id: Optional[str] = Query(None),
    save_to_patient_reports: bool = Query(False)
):
    """
    Process a document for OCR and field extraction.
    
    - **document_id**: ID of the document to process
    - **request**: Process document request with options
    - **patient_id**: Optional patient ID to associate with this document
    - **save_to_patient_reports**: Whether to save to patient reports collection
    """
    try:
        # Add processing options
        options = request.dict()
        if patient_id:
            options["patient_id"] = patient_id
        if save_to_patient_reports:
            options["save_to_patient_reports"] = True
        
        # Process document
        result = await DocumentProcessor.process_document(document_id, request.force_reprocess, options)
        
        if "error" in result:
            return ProcessingStatusResponse(
                status=StatusEnum.ERROR,
                message=result["error"],
                document_id=document_id,
                status_value=ProcessingStatusEnum.FAILED,
                progress=0.0
            )
        
        # Return response
        return ProcessingStatusResponse(
            status=StatusEnum.SUCCESS,
            message="Document processing started",
            document_id=document_id,
            status_value=ProcessingStatusEnum.OCR_PROCESSING,  # Fixed: renamed to status_value
            progress=0.1,
            details=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.post("/group/process/{group_id}", response_model=Dict[str, Any])
async def process_document_group(
    group_id: str,
    request: ProcessDocumentRequest,
    patient_id: Optional[str] = Query(None),
    save_to_patient_reports: bool = Query(False)
):
    """
    Process a document group for OCR and field extraction.
    All documents in the group are processed together as a single unit.
    
    - **group_id**: ID of the document group to process
    - **request**: Process document request with options
    - **patient_id**: Optional patient ID to associate with this document group
    - **save_to_patient_reports**: Whether to save to patient reports collection
    """
    try:
        # Add processing options
        options = request.dict()
        if patient_id:
            options["patient_id"] = patient_id
        if save_to_patient_reports:
            options["save_to_patient_reports"] = True
        
        # Process document group
        result = await DocumentProcessor.process_document_group(group_id, request.force_reprocess, options)
        
        if "error" in result:
            return {
                "status": StatusEnum.ERROR,
                "message": result["error"],
                "group_id": group_id,
                "status_value": ProcessingStatusEnum.FAILED,
                "progress": 0.0
            }
        
        # Return response
        return {
            "status": StatusEnum.SUCCESS,
            "message": "Document group processing started",
            "group_id": group_id,
            "status_value": ProcessingStatusEnum.OCR_PROCESSING,
            "progress": 0.1,
            "details": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document group: {str(e)}")


@router.get("/status/{document_id}", response_model=ProcessingStatusResponse)
async def get_document_status(document_id: str):
    """
    Get the status of a document.
    
    - **document_id**: ID of the document
    """
    try:
        # Get document status
        result = await DocumentProcessor.get_document_status(document_id)
        
        if "error" in result:
            return ProcessingStatusResponse(
                status=StatusEnum.ERROR,
                message=result["error"],
                document_id=document_id,
                status_value=ProcessingStatusEnum.FAILED,  # Fixed: renamed to status_value
                progress=0.0
            )
        
        # Map status to enum
        status_mapping = {
            "uploaded": ProcessingStatusEnum.UPLOADED,
            "processing": ProcessingStatusEnum.OCR_PROCESSING,
            "ocr_processed": ProcessingStatusEnum.OCR_COMPLETED,
            "processed": ProcessingStatusEnum.COMPLETED,
            "failed": ProcessingStatusEnum.FAILED
        }
        
        status_enum = status_mapping.get(result["status"], ProcessingStatusEnum.OCR_PROCESSING)
        
        # Calculate progress based on status
        progress_mapping = {
            ProcessingStatusEnum.UPLOADED: 0.0,
            ProcessingStatusEnum.OCR_PROCESSING: 0.3,
            ProcessingStatusEnum.OCR_COMPLETED: 0.5,
            ProcessingStatusEnum.EXTRACTION_PROCESSING: 0.7,
            ProcessingStatusEnum.EXTRACTION_COMPLETED: 0.9,
            ProcessingStatusEnum.COMPLETED: 1.0,
            ProcessingStatusEnum.FAILED: 0.0
        }
        
        progress = progress_mapping.get(status_enum, 0.0)
        
        # Return response
        return ProcessingStatusResponse(
            status=StatusEnum.SUCCESS,
            message=f"Document status: {result['status']}",
            document_id=document_id,
            status_value=status_enum,  # Fixed: renamed to status_value
            progress=progress,
            details=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document status: {str(e)}")


@router.get("/group/status/{group_id}", response_model=Dict[str, Any])
async def get_group_status(group_id: str):
    """
    Get the status of a document group.
    
    - **group_id**: ID of the document group
    """
    try:
        # Retrieve document group from database
        group_data = await document_groups_collection.find_one({"id": group_id})
        if not group_data:
            raise HTTPException(status_code=404, detail=f"Document group with ID {group_id} not found")
        
        # Convert MongoDB document to Python dict
        group_data = sanitize_mongodb_document(group_data)
        
        # Get all documents in the group
        document_ids = group_data.get("document_ids", [])
        documents = []
        
        for doc_id in document_ids:
            doc_data = await documents_collection.find_one({"id": doc_id})
            if doc_data:
                doc_data = sanitize_mongodb_document(doc_data)
                documents.append(doc_data)
        
        # Map status to enum
        status_mapping = {
            "created": ProcessingStatusEnum.UPLOADED,
            "processing": ProcessingStatusEnum.OCR_PROCESSING,
            "ocr_processed": ProcessingStatusEnum.OCR_COMPLETED,
            "processed": ProcessingStatusEnum.COMPLETED,
            "failed": ProcessingStatusEnum.FAILED
        }
        
        status_enum = status_mapping.get(group_data.get("status"), ProcessingStatusEnum.UPLOADED)
        
        # Calculate progress based on status
        progress_mapping = {
            ProcessingStatusEnum.UPLOADED: 0.0,
            ProcessingStatusEnum.OCR_PROCESSING: 0.3,
            ProcessingStatusEnum.OCR_COMPLETED: 0.5,
            ProcessingStatusEnum.EXTRACTION_PROCESSING: 0.7,
            ProcessingStatusEnum.EXTRACTION_COMPLETED: 0.9,
            ProcessingStatusEnum.COMPLETED: 1.0,
            ProcessingStatusEnum.FAILED: 0.0
        }
        
        progress = progress_mapping.get(status_enum, 0.0)
        
        # Prepare OCR result info if available
        ocr_result_info = None
        if "ocr_result_id" in group_data and group_data["ocr_result_id"]:
            ocr_result_data = await ocr_results_collection.find_one({"id": group_data["ocr_result_id"]})
            if ocr_result_data:
                ocr_result_data = sanitize_mongodb_document(ocr_result_data)
                ocr_result_info = {
                    "ocr_result_id": group_data["ocr_result_id"],
                    "text_sample": ocr_result_data.get("text", "")[:500] + ("..." if len(ocr_result_data.get("text", "")) > 500 else ""),
                    "confidence": ocr_result_data.get("confidence", 0),
                    "page_count": len(ocr_result_data.get("pages", []))
                }
        
        # Prepare TRF data info if available
        trf_data_info = None
        if "trf_data_id" in group_data and group_data["trf_data_id"]:
            trf_data = await trf_data_collection.find_one({"id": group_data["trf_data_id"]})
            if trf_data:
                trf_data = sanitize_mongodb_document(trf_data)
                trf_data_info = {
                    "trf_data_id": group_data["trf_data_id"],
                    "extraction_confidence": trf_data.get("extraction_confidence", 0),
                    "missing_required_fields": trf_data.get("missing_required_fields", []),
                    "low_confidence_fields": trf_data.get("low_confidence_fields", [])
                }
        
        # Return response
        return {
            "status": StatusEnum.SUCCESS,
            "message": f"Document group status: {group_data.get('status')}",
            "group_id": group_id,
            "group_name": group_data.get("name"),
            "status_value": status_enum,
            "progress": progress,
            "document_count": len(documents),
            "documents": documents,
            "ocr_result": ocr_result_info,
            "trf_data": trf_data_info
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document group status: {str(e)}")


@router.get("/ocr/{document_id}", response_model=OCRResultResponse)
async def get_ocr_result(document_id: str):
    """
    Get the OCR result for a document.
    
    - **document_id**: ID of the document
    """
    try:
        # Get document
        document_data = await documents_collection.find_one({"id": document_id})
        if not document_data:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
        
        document = Document(**document_data)
        
        # Check if OCR result exists
        if not document.ocr_result_id:
            raise HTTPException(status_code=404, detail=f"OCR result not found for document {document_id}")
        
        # Get OCR result
        ocr_result_data = await ocr_results_collection.find_one({"id": document.ocr_result_id})
        if not ocr_result_data:
            raise HTTPException(status_code=404, detail=f"OCR result with ID {document.ocr_result_id} not found")
        
        # Return response
        return OCRResultResponse(
            status=StatusEnum.SUCCESS,
            message="OCR result retrieved successfully",
            document_id=document_id,
            ocr_result_id=document.ocr_result_id,
            text_sample=ocr_result_data["text"][:500] + ("..." if len(ocr_result_data["text"]) > 500 else ""),
            confidence=ocr_result_data["confidence"],
            processing_time=ocr_result_data["processing_time"],
            page_count=len(ocr_result_data.get("pages", []))
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting OCR result: {str(e)}")


@router.get("/trf/{id}", response_model=TRFDataResponse)
async def get_trf_data(id: str, patient_id: Optional[str] = None):
    """
    Get the TRF data for a document or document group.
    
    - **id**: ID of the document or document group
    - **patient_id**: Optional patient ID to check for existing data
    """
    try:
        # First check if there's patient data if patient_id provided
        if patient_id:
            patient_data = await patientreports_collection.find_one({"patientID": patient_id})
            if patient_data and "trf_data_id" in patient_data:
                # This patient already has TRF data linked
                trf_data_id = patient_data["trf_data_id"]
                
                # Get the TRF data
                trf_data = await trf_data_collection.find_one({"id": trf_data_id})
                if trf_data:
                    # Sanitize the MongoDB document to make it JSON serializable
                    sanitized_trf_data = sanitize_mongodb_document(trf_data)
                    
                    # Return TRF data with patient context
                    return TRFDataResponse(
                        status=StatusEnum.SUCCESS,
                        message="TRF data retrieved successfully for patient",
                        document_id=id,
                        trf_data_id=trf_data_id,
                        trf_data=sanitized_trf_data,
                        missing_required_fields=sanitized_trf_data.get("missing_required_fields", []),
                        low_confidence_fields=sanitized_trf_data.get("low_confidence_fields", []),
                        extraction_confidence=sanitized_trf_data.get("extraction_confidence", 0.0),
                        completion_percentage=0.5  # Example value
                    )
        
        # Otherwise, proceed with regular TRF data retrieval
        result = await DocumentProcessor.get_trf_data(id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        # Calculate completion percentage
        completion_percentage = 0.0
        if "missing_required_fields" in result["trf_data"]:
            from ..schemas.trf_schema import REQUIRED_FIELDS
            missing_count = len(result["trf_data"]["missing_required_fields"])
            total_required = len(REQUIRED_FIELDS)
            completion_percentage = (total_required - missing_count) / total_required if total_required > 0 else 1.0
        
        # Return response
        return TRFDataResponse(
            status=StatusEnum.SUCCESS,
            message="TRF data retrieved successfully",
            document_id=id,  # Use the provided id regardless of whether it's a document or group
            trf_data_id=result["trf_data_id"],
            trf_data=result["trf_data"],
            missing_required_fields=result["trf_data"].get("missing_required_fields", []),
            low_confidence_fields=result["trf_data"].get("low_confidence_fields", []),
            extraction_confidence=result["trf_data"].get("extraction_confidence", 0.0),
            completion_percentage=completion_percentage
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting TRF data: {str(e)}")


@router.post("/map-to-patient", response_model=Dict[str, Any])
async def map_trf_data_to_patient(
    trf_data: Dict[str, Any] = Body(..., description="TRF data to map"),
    patient_id: Optional[str] = Body(None, description="Patient ID to associate with TRF data"),
    form_id: Optional[str] = Body(None, description="Form ID to associate with TRF data"),
    save_to_collection: bool = Body(False, description="Whether to save the mapped data to patientreports collection")
):
    try:
        # Normalize the raw data (basic structure fixes)
        trf_data = normalize_array_fields(trf_data)

        # Fetch OCR text if available
        ocr_result = None
        ocr_text = ""
        if "document_id" in trf_data:
            raw_ocr = await ocrresults_collection.find_one({"document_id": trf_data["document_id"]})
            if raw_ocr:
                ocr_result = OCRResult(**raw_ocr)
                ocr_text = ocr_result.text or ""

        # Run AI agent to autofill/correct missing or malformed fields
        if ocr_result:
            agent_response = await AgentSuggestions.generate_suggestions(ocr_result, trf_data)
            for s in agent_response.get("suggestions", []):
                if s.get("suggested_value") and s.get("confidence", 0) >= 0.7:
                    set_field_value(trf_data, s["field_path"], s["suggested_value"])

        # Add patientID if not set
        if patient_id:
            trf_data["patientID"] = patient_id

        # Add default required metadata
        if "viewOnlyMode" not in trf_data:
            trf_data["viewOnlyMode"] = False
        if "formStatus" not in trf_data:
            trf_data["formStatus"] = "Draft"
        if "initiatedDate" not in trf_data:
            now = datetime.now().isoformat()
            trf_data["initiatedDate"] = now
            trf_data["lastUpdated"] = now

        # Validate and convert to model
        trf_data = normalize_array_fields(trf_data)
        patient_report = PatientReport(**trf_data)
        patient_data = patient_report.dict()

        # Save to DB if needed
        if save_to_collection and (patient_id or form_id):
            query = {"patientID": patient_id} if patient_id else {"reportId": form_id}
            await patientreports_collection.update_one(query, {"$set": patient_data}, upsert=True)

        return {
            "status": StatusEnum.SUCCESS,
            "message": "TRF data mapped and validated successfully",
            "data": patient_data,
            "saved_to_collection": save_to_collection
        }

    except Exception as e:
        return {
            "status": StatusEnum.ERROR,
            "message": str(e),
            "data": trf_data,
            "saved_to_collection": False
        }


@router.post("/process-for-patient/{document_id}/{patient_id}", response_model=Dict[str, Any])
async def process_document_for_patient(
    document_id: str,
    patient_id: str,
    request: Request,
    force_reprocess: bool = Query(False),
    save_to_reports: bool = Query(True)
):
    try:
        # Get the request body if it exists
        request_data = await request.json() if request.headers.get("content-type") == "application/json" else {}
        
        # Extract patient data if provided
        patient_data = request_data.get("patient_data")
        
        # Setup processing options
        options = {
            "patient_id": patient_id,
            "save_to_patient_reports": save_to_reports
        }
        
        # Add patient data to options if provided
        if patient_data:
            options["patient_data"] = patient_data
            
        # Process document with patient context
        result = await DocumentProcessor.process_document(document_id, force_reprocess, options)
        
        if "error" in result:
            return {
                "status": StatusEnum.ERROR,
                "message": result["error"],
                "document_id": document_id,
                "patient_id": patient_id,
                "status_value": ProcessingStatusEnum.FAILED,
                "progress": 0.0
            }

        # 🔧 Normalize Sample, FamilyHistory.familyMember, etc.
        if "trf_data" in result:
            result["trf_data"] = normalize_array_fields(result["trf_data"])

        # Return success response
        return {
            "status": StatusEnum.SUCCESS,
            "message": f"Document processed and mapped to patient {patient_id}",
            "document_id": document_id,
            "patient_id": patient_id,
            "trf_data_id": result.get("trf_data_id"),
            "saved_to_reports": save_to_reports,
            "status_value": ProcessingStatusEnum.COMPLETED,
            "progress": 1.0,
            "details": result
        }
        
    except Exception as e:
        print(f"Error processing document for patient: {str(e)}")
        return {
            "status": StatusEnum.ERROR,
            "message": str(e),
            "document_id": document_id,
            "patient_id": patient_id,
            "status_value": ProcessingStatusEnum.FAILED,
            "progress": 0.0
        }

@router.post("/group/process-for-patient/{group_id}/{patient_id}", response_model=Dict[str, Any])
async def process_group_for_patient(
    group_id: str,
    patient_id: str,
    force_reprocess: bool = Query(False, description="Whether to force reprocessing if already processed"),
    save_to_reports: bool = Query(True, description="Whether to save to patient reports collection")
):
    """
    Process a document group for OCR and field extraction, then map to patient and save to patientreports collection.
    
    - **group_id**: ID of the document group to process
    - **patient_id**: Patient ID to associate with this document group
    - **force_reprocess**: Whether to force reprocessing if already processed
    - **save_to_reports**: Whether to save to patient reports collection
    """
    try:
        # First, process the document group
        options = {
            "patient_id": patient_id,
            "save_to_patient_reports": save_to_reports
        }
        
        # Create request object
        from ..schemas.request_schemas import ProcessDocumentRequest
        request = ProcessDocumentRequest(force_reprocess=force_reprocess)
        
        # Process document group
        result = await DocumentProcessor.process_document_group(group_id, force_reprocess, options)
        
        if "error" in result:
            return {
                "status": StatusEnum.ERROR,
                "message": result["error"],
                "group_id": group_id,
                "patient_id": patient_id,
                "status_value": ProcessingStatusEnum.FAILED,
                "progress": 0.0
            }
        
        # Return response with patient context
        return {
            "status": StatusEnum.SUCCESS,
            "message": f"Document group processed and mapped to patient {patient_id}",
            "group_id": group_id,
            "patient_id": patient_id,
            "trf_data_id": result.get("trf_data_id"),
            "saved_to_reports": save_to_reports,
            "status_value": ProcessingStatusEnum.COMPLETED,
            "progress": 1.0,
            "details": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document group for patient: {str(e)}")

@router.get("/list", response_model=Dict[str, Any])
async def list_documents(
    limit: int = Query(10, ge=1, le=100),
    skip: int = Query(0, ge=0),
    status: Optional[str] = Query(None)
):
    """
    List documents with optional filtering.
    
    - **limit**: Maximum number of documents to return
    - **skip**: Number of documents to skip
    - **status**: Filter by document status
    """
    try:
        # List documents
        result = await DocumentProcessor.list_documents(limit, skip, status)
        
        if "error" in result:
            return {"status": StatusEnum.ERROR, "message": result["error"]}
        
        # Return response
        return {
            "status": StatusEnum.SUCCESS,
            "message": f"Retrieved {len(result['documents'])} documents",
            "total": result["total"],
            "limit": result["limit"],
            "skip": result["skip"],
            "documents": result["documents"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@router.get("/group/list", response_model=Dict[str, Any])
async def list_document_groups(
    limit: int = Query(10, ge=1, le=100),
    skip: int = Query(0, ge=0),
    status: Optional[str] = Query(None)
):
    """
    List document groups with optional filtering.
    
    - **limit**: Maximum number of groups to return
    - **skip**: Number of groups to skip
    - **status**: Filter by group status
    """
    try:
        # Prepare filter
        filter_dict = {}
        if status:
            filter_dict["status"] = status
        
        # Get total count
        total = await document_groups_collection.count_documents(filter_dict)
        
        # Get document groups
        cursor = document_groups_collection.find(filter_dict).skip(skip).limit(limit)
        groups = []
        
        async for group in cursor:
            # Sanitize document to ensure it's JSON serializable
            group = sanitize_mongodb_document(group)
            
            # Get document count
            document_ids = group.get("document_ids", [])
            
            # Add group to list with document count
            groups.append({
                **group,
                "document_count": len(document_ids)
            })
        
        # Return response
        return {
            "status": StatusEnum.SUCCESS,
            "message": f"Retrieved {len(groups)} document groups",
            "total": total,
            "limit": limit,
            "skip": skip,
            "groups": groups
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing document groups: {str(e)}")


@router.put("/trf/{document_id}/field", response_model=Dict[str, Any])
async def update_trf_field(
    document_id: str,
    field_path: str = Query(..., description="Path to the field to update"),
    field_value: str = Query(..., description="New value for the field"),
    confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Confidence score for the update")
):
    """
    Update a field in the TRF data.
    
    - **document_id**: ID of the document
    - **field_path**: Path to the field to update
    - **field_value**: New value for the field
    - **confidence**: Confidence score for the update
    """
    try:
        # Update field
        result = await DocumentProcessor.update_trf_field(document_id, field_path, field_value, confidence)
        
        if "error" in result:
            return {"status": StatusEnum.ERROR, "message": result["error"]}
        
        # Return response
        return {
            "status": StatusEnum.SUCCESS,
            "message": f"Field '{field_path}' updated successfully",
            "document_id": document_id,
            "field_path": field_path,
            "previous_value": result["previous_value"],
            "new_value": result["new_value"],
            "confidence": result["confidence"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating TRF field: {str(e)}")

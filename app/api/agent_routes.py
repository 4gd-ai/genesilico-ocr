"""API routes for AI agent interactions."""

import time
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Path, Query, HTTPException, Body
from fastapi.responses import JSONResponse

from ..core.database import documents_collection, ocr_results_collection, trf_data_collection
from ..models.document import Document, OCRResult
from ..models.trf import PatientReport
from ..agent.reasoning import agent_reasoning
from ..agent.suggestions import AgentSuggestions
from ..schemas.request_schemas import AgentQueryRequest, FieldUpdateRequest
from ..schemas.response_schemas import AgentQueryResponse, StatusEnum


router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.post("/query/{document_id}", response_model=AgentQueryResponse)
async def query_agent(
    document_id: str,
    request: AgentQueryRequest
):
    """
    Query the AI agent about a document.
    
    - **document_id**: ID of the document
    - **request**: Agent query request with question and context
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
        
        ocr_result = OCRResult(**ocr_result_data)
        
        # Get TRF data if it exists
        trf_data = {}
        if document.trf_data_id:
            trf_data_doc = await trf_data_collection.find_one({"id": document.trf_data_id})
            if trf_data_doc:
                trf_data = trf_data_doc
        
        # Query agent
        response = await agent_reasoning.query_agent(request.query, ocr_result.text, trf_data)
        
        # Return response
        return AgentQueryResponse(
            status=StatusEnum.SUCCESS,
            message="Agent query successful",
            document_id=document_id,
            query=request.query,
            response=response["response"],
            suggested_actions=response.get("suggested_actions", [])
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying agent: {str(e)}")


@router.get("/suggestions/{document_id}", response_model=Dict[str, Any])
async def get_suggestions(document_id: str):
    """
    Get AI agent suggestions for a document.
    
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
        
        ocr_result = OCRResult(**ocr_result_data)
        
        # Get TRF data if it exists
        trf_data = {}
        if document.trf_data_id:
            trf_data_doc = await trf_data_collection.find_one({"id": document.trf_data_id})
            if trf_data_doc:
                trf_data = trf_data_doc
        
        # Generate suggestions
        suggestions = await AgentSuggestions.generate_suggestions(ocr_result, trf_data)
        
        # Return response
        return {
            "status": StatusEnum.SUCCESS,
            "message": "Suggestions generated successfully",
            "document_id": document_id,
            "suggestions": suggestions["suggestions"],
            "missing_fields": suggestions["missing_fields"],
            "low_confidence_fields": suggestions["low_confidence_fields"],
            "completion_percentage": suggestions["completion_percentage"]
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating suggestions: {str(e)}")


@router.get("/suggestions/{document_id}/field", response_model=Dict[str, Any])
async def get_field_suggestion(
    document_id: str,
    field_path: str = Query(..., description="Path to the field")
):
    """
    Get AI agent suggestion for a specific field.
    
    - **document_id**: ID of the document
    - **field_path**: Path to the field
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
        
        # Get TRF data if it exists
        trf_data = {}
        if document.trf_data_id:
            trf_data_doc = await trf_data_collection.find_one({"id": document.trf_data_id})
            if trf_data_doc:
                trf_data = trf_data_doc
        
        # Get field suggestion
        suggestion = await AgentSuggestions.get_field_suggestions(
            field_path, 
            ocr_result_data["text"], 
            trf_data
        )
        
        # Return response
        return {
            "status": StatusEnum.SUCCESS,
            "message": f"Suggestion for field '{field_path}' generated successfully",
            "document_id": document_id,
            "field_path": field_path,
            "suggestion": suggestion
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating field suggestion: {str(e)}")


@router.get("/suggestions/{document_id}/missing", response_model=Dict[str, Any])
async def get_missing_field_suggestions(document_id: str):
    """
    Get AI agent suggestions for all missing required fields.
    
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
        
        # Get TRF data if it exists
        trf_data = {}
        if document.trf_data_id:
            trf_data_doc = await trf_data_collection.find_one({"id": document.trf_data_id})
            if trf_data_doc:
                trf_data = trf_data_doc
        
        # Get missing field suggestions
        suggestions = await AgentSuggestions.get_missing_field_suggestions(
            ocr_result_data["text"], 
            trf_data
        )
        
        # Return response
        return {
            "status": StatusEnum.SUCCESS,
            "message": "Missing field suggestions generated successfully",
            "document_id": document_id,
            "missing_fields": suggestions["missing_fields"],
            "suggestions": suggestions["suggestions"]
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating missing field suggestions: {str(e)}")


@router.get("/completion/{document_id}", response_model=Dict[str, Any])
async def get_completion_guidance(document_id: str):
    """
    Get guidance for completing the TRF form.
    
    - **document_id**: ID of the document
    """
    try:
        # Get document
        document_data = await documents_collection.find_one({"id": document_id})
        if not document_data:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
        
        document = Document(**document_data)
        
        # Get TRF data if it exists
        trf_data = {}
        if document.trf_data_id:
            trf_data_doc = await trf_data_collection.find_one({"id": document.trf_data_id})
            if trf_data_doc:
                trf_data = trf_data_doc
        
        # Get completion guidance
        guidance = await AgentSuggestions.get_completion_guidance(trf_data)
        
        # Return response
        return {
            "status": StatusEnum.SUCCESS,
            "message": "Completion guidance generated successfully",
            "document_id": document_id,
            "completion_percentage": guidance["completion_percentage"],
            "missing_fields": guidance["missing_fields"],
            "guidance_message": guidance["guidance_message"]
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating completion guidance: {str(e)}")

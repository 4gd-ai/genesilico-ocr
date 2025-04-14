"""Main API router that includes all route modules."""

from fastapi import APIRouter

from .document_routes import router as document_router
from .agent_routes import router as agent_router


# Main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(document_router)
api_router.include_router(agent_router)

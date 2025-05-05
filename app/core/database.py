import motor.motor_asyncio
from pymongo import MongoClient
from ..config import settings

# MongoDB connection string
MONGODB_URL = settings.MONGODB_URL

# Database name
DB_NAME = settings.MONGODB_DB

# Async MongoDB client for FastAPI
async_client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
async_db = async_client[DB_NAME]

# Sync MongoDB client for utility functions
sync_client = MongoClient(MONGODB_URL)
sync_db = sync_client[DB_NAME]

# Collections
documents_collection = async_db.documents_collection
document_groups_collection = async_db.document_groups_collection
ocr_results_collection = async_db.ocr_results_collection
trf_data_collection = async_db.trf_data_collection
patientreports_collection = async_db.patientreports_collection


async def connect_to_mongodb():
    """Connect to MongoDB."""
    try:
        # Trigger connection verification
        await async_client.admin.command('ping')
        print(f"Connected to MongoDB at {MONGODB_URL}")
        return True
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return False


async def close_mongodb_connection():
    """Close MongoDB connection."""
    try:
        async_client.close()
        sync_client.close()
        print("MongoDB connection closed")
    except Exception as e:
        print(f"Error closing MongoDB connection: {e}")

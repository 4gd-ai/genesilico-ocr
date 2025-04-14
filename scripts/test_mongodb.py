#!/usr/bin/env python
"""Script to test MongoDB connection."""

import os
import sys
import asyncio
import argparse
from pathlib import Path
import motor.motor_asyncio
from pymongo import MongoClient
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()


async def test_async_connection(url, db_name):
    """Test async MongoDB connection."""
    try:
        # Create client
        client = motor.motor_asyncio.AsyncIOMotorClient(url)
        
        # Get database
        db = client[db_name]
        
        # Test connection
        result = await db.command("ping")
        
        if result.get("ok", 0) == 1:
            print(f"Async MongoDB connection successful to {url}/{db_name}")
            return True
        else:
            print(f"Async MongoDB connection failed: {result}")
            return False
    except Exception as e:
        print(f"Async MongoDB connection error: {e}")
        return False
    finally:
        # Close client
        if 'client' in locals():
            client.close()


def test_sync_connection(url, db_name):
    """Test sync MongoDB connection."""
    try:
        # Create client
        client = MongoClient(url)
        
        # Get database
        db = client[db_name]
        
        # Test connection
        result = db.command("ping")
        
        if result.get("ok", 0) == 1:
            print(f"Sync MongoDB connection successful to {url}/{db_name}")
            return True
        else:
            print(f"Sync MongoDB connection failed: {result}")
            return False
    except Exception as e:
        print(f"Sync MongoDB connection error: {e}")
        return False
    finally:
        # Close client
        if 'client' in locals():
            client.close()


async def main():
    """Test MongoDB connection."""
    parser = argparse.ArgumentParser(description="Test MongoDB connection")
    parser.add_argument("--url", help="MongoDB URL", default=os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    parser.add_argument("--db", help="MongoDB database name", default=os.getenv("MONGODB_DB", "genesilico_ocr"))
    args = parser.parse_args()
    
    # Test sync connection
    sync_success = test_sync_connection(args.url, args.db)
    
    # Test async connection
    async_success = await test_async_connection(args.url, args.db)
    
    if sync_success and async_success:
        print("All MongoDB connection tests passed")
        return 0
    else:
        print("Some MongoDB connection tests failed")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))

from pymongo import MongoClient, AsyncMongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from app.config import settings

_client: MongoClient | None = None
_async_client: AsyncMongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.mongodb_uri)
    return _client


def get_async_client() -> AsyncMongoClient:
    global _async_client
    if _async_client is None:
        _async_client = AsyncMongoClient(settings.mongodb_uri)
    return _async_client


def get_database() -> Database:
    return get_client()[settings.mongodb_db_name]


def get_collection() -> Collection:
    return get_database()[settings.mongodb_collection_name]


def ping() -> bool:
    try:
        get_client().admin.command("ping")
        return True
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return False
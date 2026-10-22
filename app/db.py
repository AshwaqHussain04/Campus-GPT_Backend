from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId
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

def get_collection_admin() -> Collection:
    return get_database()[settings.mongodb_collection_admin]

def get_collection_student() -> Collection:
    return get_database()[settings.mongodb_collection_student]

def get_collection_history() -> Collection:
    return get_database()[settings.mongodb_collection_history]

def create_student(regNo: int, email: str, hashed_password: str, name: str) -> str:
    result = get_collection_student().insert_one({
        "email": email,
        "name": name,
        "regNo": regNo,
        "hashed_password": hashed_password,
        "created_at": datetime.now(timezone.utc),
    })
    return str(result.inserted_id)


def get_student_by_email(email: str) -> dict | None:
    return get_collection_student().find_one(
        {"email": email},
        {"hashed_password": 0},
    )


def get_student_for_login(email: str) -> dict | None:
    return get_collection_student().find_one({"email": email})


def create_admin(name: str, hashed_password: str, email: str) -> str:
    result = get_collection_admin().insert_one({
        "username": name,
        "email" : email,
        "hashed_password": hashed_password,
        "created_at": datetime.now(timezone.utc),
    })
    return str(result.inserted_id)


def get_admin_by_username(username: str) -> dict | None:
    return get_collection_admin().find_one(
        {"username": username},
        {"hashed_password": 0},
    )

def create_chat_pending(email: str, query: str) -> str:
    now = datetime.now(timezone.utc)
    result = get_collection_history().insert_one({
        "email": email,
        "query" : query,
        "answer": None,
        "status" : "pending",
        "error": None,
        "created_at": now,
        "updated_at": now,
    })
    return str(result.inserted_id)

def update_chat_success(chat_id: str, email: str, answer: str) -> None:
    get_collection_history().update_one(
        {"_id": ObjectId(chat_id), "email": email},
        {"$set": {
            "answer": answer,
            "status": "completed",
            "error": None,
            "updated_at": datetime.now(timezone.utc),
        }},
    )


def update_chat_failed(chat_id: str, email: str, error_message: str) -> None:
    get_collection_history().update_one(
        {"_id": ObjectId(chat_id), "email": email},
        {"$set": {
            "answer": None,
            "status": "failed",
            "error": error_message,
            "updated_at": datetime.now(timezone.utc),
        }},
    )


def get_chat_history(email: str, limit: int = 20) -> list[dict]:
    return list(
        get_collection_history()
        .find({"email": email})
        .sort("created_at", -1)
        .limit(limit)
    )


def delete_chat_history_item(email: str, chat_id: str) -> bool:
    try:
        object_id = ObjectId(chat_id)
    except InvalidId:
        return False

    result = get_collection_history().delete_one({
        "_id": object_id,
        "email": email,
    })
    return result.deleted_count == 1


def ensure_history_indexes() -> None:
    get_collection_history().create_index([("email", 1), ("created_at", -1)])


def get_admin_for_login(username: str) -> dict | None:
    return get_collection_admin().find_one({"username": username})


def ensure_user_indexes() -> None:
    get_collection_student().create_index("email", unique=True)
    get_collection_admin().create_index("username", unique=True)
    ensure_history_indexes()

def ping() -> bool:
    try:
        get_client().admin.command("ping")
        return True
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return False
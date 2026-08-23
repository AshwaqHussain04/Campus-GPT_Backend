from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    #For Testing Purposes
    mock_mode: bool = False

    # Gemini
    gemini_api_key: str

    # MongoDB
    mongodb_uri: str
    mongodb_db_name: str = "Campus-GPT_Data"
    mongodb_collection_name: str = "Study_Notes"
    mongodb_vector_index_name: str = "autoembed_index"

    # Embedding / chunking
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768    
    llm_model: str = "gemini-flash-latest"   # was "models/gemini-1.5-flash"
    chunk_size: int = 1024   # was 512
    chunk_overlap: int = 100  # was 50

    # Auth
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    admin_username: str
    admin_password_hash: str

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
        
    )


settings = Settings()
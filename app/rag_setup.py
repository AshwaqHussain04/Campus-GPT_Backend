from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.settings import Settings as LlamaSettings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.groq import Groq
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from app.config import settings
from app.db import get_async_client, get_client

STORAGE_DIR = "app/storage"


def configure_llama_settings() -> None:
    LlamaSettings.embed_model = GoogleGenAIEmbedding(
    model_name=settings.embedding_model,
    api_key=settings.gemini_api_key,
    embedding_config={"output_dimensionality": settings.embedding_dimensions},
    embed_batch_size=50,   # fewer, larger requests instead of one-per-chunk
)
    LlamaSettings.llm = Groq(
    model=settings.llm_model,  # or a smaller/faster variant
    api_key=settings.grok_api_key,
    )
    LlamaSettings.node_parser = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def get_vector_store() -> MongoDBAtlasVectorSearch:
    return MongoDBAtlasVectorSearch(
        mongodb_client=get_client(),
        async_mongodb_client=get_async_client(),
        db_name=settings.mongodb_db_name,
        collection_name=settings.mongodb_collection_name,
        vector_index_name=settings.mongodb_vector_index_name,
    )


def get_hybrid_retriever(similarity_top_k: int = 5) -> QueryFusionRetriever:
    """
    Combines dense vector search (MongoDB Atlas) with BM25 keyword search
    (local docstore) using reciprocal rank fusion.
    """
    docstore = SimpleDocumentStore.from_persist_dir(STORAGE_DIR)
    vector_store = get_vector_store()

    vector_index = VectorStoreIndex.from_vector_store(vector_store)
    vector_retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)

    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=similarity_top_k,
    )

    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=similarity_top_k,
        num_queries=1,  # 1 = don't generate query variations, just fuse the two retrievers as-is
        mode="reciprocal_rerank",
        use_async=False,
    )
import shutil
import os

from pathlib import Path

from fastapi import Depends, File, UploadFile

from app.auth import authenticate_admin, create_access_token, get_current_admin
from app.ingest import (
    find_new_or_changed_files,
    ingest_with_retry,
    load_docstore,
    load_documents_manually,
    load_manifest,
    save_manifest,
    DATA_DIR,
    STORAGE_DIR,
)
from llama_index.core import StorageContext


from contextlib import asynccontextmanager
from app.rag_setup import configure_llama_settings, get_hybrid_retriever, get_vector_store
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.settings import Settings as LlamaSettings
from pydantic import BaseModel
from llama_index.core.query_engine import RetrieverQueryEngine
from app.rag_setup import configure_llama_settings, get_hybrid_retriever
from app.db import ping
from app.rag_setup import configure_llama_settings

query_engine = None

# Cosine similarity score below this = "not a good enough match", triggers fallback.
# Tune this after testing real queries against your data.
FALLBACK_THRESHOLD = 0.65

REFUSAL_PHRASES = [
    "does not contain",
    "doesn't contain",
    "no information",
    "not mentioned",
    "cannot find",
    "not provided in the",
    "is not available",
    "not available in",
    "no relevant information",
    "does not include",
    "don't have information",
    "do not have information",
    "not covered in",
]

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UploadResponse(BaseModel):
    filename: str
    chunks_ingested: bool
    message: str



def looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not ping():
        raise RuntimeError("Could not connect to MongoDB on startup. Check MONGODB_URI in .env.local")

    configure_llama_settings()

    global query_engine
    retriever = get_hybrid_retriever(similarity_top_k=5)
    query_engine = RetrieverQueryEngine.from_args(retriever)

    yield


app = FastAPI(title="CampusGPT API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    source: str  # "campus_documents" or "general_knowledge"
    sources_used: list[str] = []

#route to check if the server is running and healthy. Returns a simple JSON response with status "ok".
@app.get("/health")
def health():
    return {"status": "ok"}

#route for sending questions to the RAG system and getting answers back, along with the source of the answer (either "campus_documents" or "general_knowledge").
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Query engine not ready yet")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    response = query_engine.query(question)
    source_nodes = response.source_nodes
    answer_text = str(response)

    print(f"DEBUG: {len(source_nodes)} source nodes retrieved")
    for i, node in enumerate(source_nodes):
        print(f"  [{i}] score={node.score}, file={node.node.metadata.get('file_name')}, text_preview={node.node.text[:80]!r}")
        print(f"DEBUG: is_refusal={looks_like_refusal(answer_text)}")

    has_any_match = len(source_nodes) > 0
    is_refusal = looks_like_refusal(answer_text)

    if has_any_match and not is_refusal:
        file_names = sorted({
            node.node.metadata.get("file_name", "unknown") for node in source_nodes
        })
        return QueryResponse(
            answer=answer_text,
            source="campus_documents",
            sources_used=file_names,
        )

    fallback = LlamaSettings.llm.complete(
        f"Answer this question using your general knowledge: {question}"
    )
    return QueryResponse(
        answer=f"Note: this is not from official campus documents.\n\n{fallback.text}",
        source="general_knowledge",
        sources_used=[],
    )


#Route for admin login. Accepts a username and password, verifies them against the stored admin credentials, and returns a JWT access token if successful.
@app.post("/auth/login", response_model=LoginResponse)
def login(request: LoginRequest):
    if not authenticate_admin(request.username, request.password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(request.username)
    return LoginResponse(access_token=token)


# route for admin to upload new documents (PDF or TXT) to the server, which will then be ingested and made available for querying.
@app.post("/admin/upload", response_model=UploadResponse)
def upload_document(
    file: UploadFile = File(...),
    admin: str = Depends(get_current_admin),
):
    if not (file.filename.lower().endswith(".pdf") or file.filename.lower().endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")

    dest_path = Path(DATA_DIR) / file.filename
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run the same incremental ingestion logic as the CLI script
    manifest = load_manifest()
    paths_to_ingest, updated_manifest = find_new_or_changed_files(manifest)

    normalized_paths = [os.path.normpath(p) for p in paths_to_ingest]
    if os.path.normpath(str(dest_path)) not in normalized_paths:
        return UploadResponse(
            filename=file.filename,
            chunks_ingested=False,
            message="File uploaded, but content is identical to an already-ingested version. Skipped.",
        )

    documents = load_documents_manually(paths_to_ingest)
    docstore = load_docstore()
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)

    ingest_with_retry(documents, storage_context)

    os.makedirs(STORAGE_DIR, exist_ok=True)
    docstore.persist(persist_path=os.path.join(STORAGE_DIR, "docstore.json"))
    save_manifest(updated_manifest)

    # Refresh the live retriever so the new content is queryable without restarting the server
    global query_engine
    retriever = get_hybrid_retriever(similarity_top_k=5)
    query_engine = RetrieverQueryEngine.from_args(retriever)

    return UploadResponse(
        filename=file.filename,
        chunks_ingested=True,
        message=f"Uploaded and ingested {len(paths_to_ingest)} new/changed file(s).",
    )

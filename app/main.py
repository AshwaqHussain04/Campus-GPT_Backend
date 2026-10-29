import os
import logging
import tempfile
from datetime import datetime
from app.config import settings
from app.mock_data import get_mock_query_response, MOCK_UPLOAD_RESPONSE

from pathlib import Path

from fastapi import Depends, File, Query, UploadFile
from pymongo.errors import DuplicateKeyError

from app.auth import authenticate_admin, authenticate_student, create_access_token, get_current_admin, get_current_student, hash_password, verify_admin_secret
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
from app.documents import delete_document as remove_document, document_stats, list_documents, resolve_document_path
from llama_index.core import StorageContext


from contextlib import asynccontextmanager
from app.rag_setup import configure_llama_settings, get_hybrid_retriever, get_vector_store
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.settings import Settings as LlamaSettings
from pydantic import BaseModel
from llama_index.core.query_engine import RetrieverQueryEngine
from app.rag_setup import configure_llama_settings, get_hybrid_retriever
from app.db import create_admin, create_chat_pending, create_student, delete_chat_history_item, ensure_user_indexes, get_chat_history, ping, update_chat_failed, update_chat_success
from app.rag_setup import configure_llama_settings

query_engine = None
logger = logging.getLogger(__name__)
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024

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

class SignupRequest_Student(BaseModel):
    regNo: int
    password: str
    email: str
    name: str

class SignupRequest_Admin(BaseModel):
    name: str
    password: str
    email: str
    secret_key: str

class SignupResponse_Admin(BaseModel):
    access_token: str
    name: str
    token_type: str = "bearer"   

class SignupResponse_Student(BaseModel):
    access_token: str
    name: str
    token_type: str = "bearer"   

class LoginRequest_Admin(BaseModel):
    username: str
    password: str
    secret_key: str

class LoginRequest_Student(BaseModel):
    email: str
    password: str

class LoginResponse_Admin(BaseModel):
    access_token: str
    name: str
    token_type: str = "bearer"

class LoginResponse_Student(BaseModel):
    access_token: str
    name: str
    token_type: str = "bearer"



class UploadResponse(BaseModel):
    filename: str
    chunks_ingested: bool
    message: str


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    type: str
    size_bytes: int
    uploaded_at: datetime
    status: str


class DocumentStats(BaseModel):
    total_documents: int
    ready_documents: int
    not_indexed_documents: int
    total_size_bytes: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]
    stats: DocumentStats


class DocumentDeleteResponse(BaseModel):
    document_id: str
    message: str



def looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.mock_mode:
        print("Running in MOCK_MODE — skipping MongoDB connection and retriever setup.")
        yield
        return
    
    if not ping():
        raise RuntimeError("Could not connect to MongoDB on startup. Check MONGODB_URI in .env.local")

    ensure_user_indexes()

    configure_llama_settings()

    global query_engine
    retriever = get_hybrid_retriever(similarity_top_k=5)
    query_engine = RetrieverQueryEngine.from_args(retriever)

    yield


app = FastAPI(title="CampusGPT API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
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


class ChatHistoryRecord(BaseModel):
    id: str
    query: str
    answer: str | None = None
    status: str
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class ChatHistoryResponse(BaseModel):
    items: list[ChatHistoryRecord]


class ChatHistoryDeleteResponse(BaseModel):
    id: str
    message: str

#6: route to check if the server is running and healthy. Returns a simple JSON response with status "ok".
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/chat/history", response_model=ChatHistoryResponse)
def chat_history(
    limit: int = Query(default=50, ge=1, le=100),
    student: str = Depends(get_current_student),
):
    if settings.mock_mode:
        return ChatHistoryResponse(items=[])

    records = get_chat_history(student, limit=limit)
    return ChatHistoryResponse(items=[
        ChatHistoryRecord(
            id=str(record["_id"]),
            query=record["query"],
            answer=record.get("answer"),
            status=record["status"],
            error=record.get("error"),
            created_at=record["created_at"],
            updated_at=record["updated_at"],
        )
        for record in records
    ])


@app.delete("/chat/history/{chat_id}", response_model=ChatHistoryDeleteResponse)
def delete_student_chat_history(
    chat_id: str,
    student: str = Depends(get_current_student),
):
    if settings.mock_mode or not delete_chat_history_item(student, chat_id):
        raise HTTPException(status_code=404, detail="Chat history item not found.")
    return ChatHistoryDeleteResponse(id=chat_id, message="Chat history item deleted.")

#1: route for sending questions to the RAG system and getting answers back, along with the source of the answer (either "campus_documents" or "general_knowledge").
@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    student: str = Depends(get_current_student),
    ):

    if(student is None):
        raise HTTPException(status_code=401, detail="Proper authentication required")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if settings.mock_mode:
        return QueryResponse(**get_mock_query_response(question))

    if query_engine is None:
        raise HTTPException(status_code=503, detail="Query engine not ready yet")

    chat_id = None
    try:
        chat_id = create_chat_pending(student, question)

        if query_engine is None:
            raise HTTPException(status_code=503, detail="Query engine not ready yet")

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
            update_chat_success(chat_id, student, answer_text)
            return QueryResponse(
                answer=answer_text,
                source="campus_documents",
                sources_used=file_names,
            )

        fallback = LlamaSettings.llm.complete(
            f"Answer this question using your general knowledge: {question}"
        )
        fallback_text = f"Note: this is not from official campus documents.\n\n{fallback.text}"
        update_chat_success(chat_id, student, fallback_text)
        return QueryResponse(
            answer=fallback_text,
            source="general_knowledge",
            sources_used=[],
        )
        

    except HTTPException:
        if chat_id is not None:
            update_chat_failed(chat_id, student, "The query could not be completed.")
        raise
    except Exception as error:
        logger.exception("Query failed for student %s", student)
        if chat_id is not None:
            update_chat_failed(chat_id, student, "The query could not be completed.")
        if getattr(error, "code", None) == 503:
            raise HTTPException(
                status_code=503,
                detail="The AI service is temporarily unavailable. Please try again shortly.",
            )
        raise HTTPException(
            status_code=500,
            detail="Sorry, I couldn't answer that right now."
        )

#2: Route for student signup. Accepts an email, password, and name, creates a new student record in the database, and returns a JWT access token if successful.
@app.post("/auth/signupStudent", response_model=SignupResponse_Student)
def signup(request: SignupRequest_Student):
    try:
        create_student(
            request.regNo,
            request.email,
            hash_password(request.password),
            request.name,
        )
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="A student with that email already exists")
    token = create_access_token(request.email, "student")
    return SignupResponse_Student(access_token=token, name=request.name)

#3: Route for admin signup. Creates an admin account only for users with the configured teacher secret.
@app.post("/auth/signupAdmin", response_model=SignupResponse_Admin)
def signup_admin(
    request: SignupRequest_Admin,
):
    if not verify_admin_secret(request.secret_key):
        raise HTTPException(status_code=403, detail="Invalid teacher secret key")

    try:
        create_admin(request.name, hash_password(request.password), request.email)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="That admin username already exists")
    token = create_access_token(request.name, "admin")
    return SignupResponse_Admin(access_token=token, name=request.name)


#4: Route for admin login. Accepts a username and password, verifies them against the stored admin credentials, and returns a JWT access token if successful.
@app.post("/auth/loginAdmin", response_model=LoginResponse_Admin)
def login(request: LoginRequest_Admin):
    admin_user = authenticate_admin(request.username, request.password, request.secret_key)
    if not admin_user:
        raise HTTPException(status_code=401, detail="Incorrect username, password, or teacher secret key")
    admin_name = admin_user.get("username") if isinstance(admin_user, dict) else request.username
    token = create_access_token(request.username, "admin")
    return LoginResponse_Admin(access_token=token, name=admin_name)

#5: Route for student Login. Accepts an email and password *TO ADD REGISTRATION NUMBER CHECK
@app.post("/auth/loginStudent", response_model=LoginResponse_Student)
def login(request: LoginRequest_Student):
    student_user = authenticate_student(request.email, request.password)
    if not student_user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    student_name = student_user.get("name") if isinstance(student_user, dict) else request.email
    token = create_access_token(request.email, "student")
    return LoginResponse_Student(access_token=token, name=student_name)


#6: route for admin to upload new documents (PDF or TXT) to the server, which will then be ingested and made available for querying.
@app.get("/admin/documents", response_model=DocumentListResponse)
def get_admin_documents(admin: object = Depends(get_current_admin)):
    documents = [] if settings.mock_mode else list_documents()
    return DocumentListResponse(documents=documents, stats=document_stats(documents))


@app.get("/admin/documents/stats", response_model=DocumentStats)
def get_admin_document_stats(admin: object = Depends(get_current_admin)):
    documents = [] if settings.mock_mode else list_documents()
    return DocumentStats(**document_stats(documents))


@app.delete("/admin/documents", response_model=DocumentDeleteResponse)
def delete_admin_document(
    document_id: str = Query(..., min_length=1),
    admin: object = Depends(get_current_admin),
):
    if settings.mock_mode:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        target = resolve_document_path(document_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Document not found.")

    docstore = load_docstore()
    vector_store = get_vector_store()
    try:
        remove_document(document_id, docstore, vector_store)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        logger.exception("Failed to delete document %s", document_id)
        raise HTTPException(status_code=500, detail="The document could not be deleted cleanly.") from error

    global query_engine
    try:
        retriever = get_hybrid_retriever(similarity_top_k=5)
        query_engine = RetrieverQueryEngine.from_args(retriever)
    except Exception as error:
        logger.exception("Document deleted, but the query engine could not be refreshed")
        query_engine = None
        raise HTTPException(status_code=503, detail="The document was deleted, but Campus-GPT needs to reload its knowledge index.") from error

    return DocumentDeleteResponse(document_id=document_id, message="Document deleted from campus knowledge.")


@app.post("/admin/upload", response_model=UploadResponse)
def upload_document(
    file: UploadFile = File(...),
    admin: object = Depends(get_current_admin),
):
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="A valid filename is required")
    if not (filename.lower().endswith(".pdf") or filename.lower().endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")
    
    if settings.mock_mode:
        return UploadResponse(**{**MOCK_UPLOAD_RESPONSE, "filename": filename})

    os.makedirs(DATA_DIR, exist_ok=True)
    dest_path = Path(DATA_DIR) / filename
    temporary_path = None
    uploaded_size = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=DATA_DIR,
            prefix=".upload-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            while chunk := file.file.read(1024 * 1024):
                uploaded_size += len(chunk)
                if uploaded_size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="The maximum supported document size is 20 MB.",
                    )
                temporary_file.write(chunk)

        if uploaded_size == 0:
            raise HTTPException(status_code=400, detail="The selected file is empty.")

        os.replace(temporary_path, dest_path)
    except Exception:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise

    # Run the same incremental ingestion logic as the CLI script
    manifest = load_manifest()
    docstore = load_docstore()
    ref_doc_infos = docstore.get_all_ref_doc_info() or {}
    indexed_file_names = {
        str((getattr(info, "metadata", {}) or {}).get("file_name", "")).casefold()
        for info in ref_doc_infos.values()
    }
    if filename.casefold() not in indexed_file_names:
        target_path = os.path.normcase(os.path.abspath(dest_path))
        manifest = {
            path: digest
            for path, digest in manifest.items()
            if os.path.normcase(os.path.abspath(path)) != target_path
        }

    paths_to_ingest, updated_manifest = find_new_or_changed_files(manifest)

    normalized_paths = [os.path.normpath(p) for p in paths_to_ingest]
    if os.path.normpath(str(dest_path)) not in normalized_paths:
        return UploadResponse(
            filename=filename,
            chunks_ingested=False,
            message="File uploaded, but content is identical to an already-ingested version. Skipped.",
        )

    documents = []
    uploaded_path = os.path.normcase(os.path.abspath(dest_path))
    for path in paths_to_ingest:
        try:
            documents.extend(load_documents_manually([path]))
        except ValueError as error:
            if os.path.normcase(os.path.abspath(path)) == uploaded_path:
                raise HTTPException(status_code=422, detail=str(error)) from error
            logger.warning("Skipping unindexable existing document %s: %s", path, error)

    if not documents:
        raise HTTPException(status_code=422, detail="No searchable text was found in the uploaded document.")

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
        filename=filename,
        chunks_ingested=True,
        message=f"Uploaded and ingested {len(paths_to_ingest)} new/changed file(s).",
    )

import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from app.ingest import file_hash, load_docstore

DATA_DIR = Path("app/data")
STORAGE_DIR = Path("app/storage")
MANIFEST_PATH = STORAGE_DIR / "manifest.json"
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def normalize_path(value: str | Path) -> str:
    normalized = str(value).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.casefold()


def resolve_document_path(document_id: str, data_dir: str | Path = DATA_DIR) -> Path:
    if not document_id or "\x00" in document_id:
        raise ValueError("A valid document ID is required.")

    root = Path(data_dir).resolve()
    candidate = (root / PurePosixPath(document_id.replace("\\", "/"))).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("The document ID is outside the knowledge directory.") from error

    if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF and TXT documents are supported.")
    return candidate


def list_documents(
    data_dir: str | Path = DATA_DIR,
    manifest_path: str | Path = MANIFEST_PATH,
    indexed_file_names: set[str] | None = None,
) -> list[dict]:
    root = Path(data_dir)
    manifest = _read_manifest(manifest_path)
    normalized_manifest = {normalize_path(path): digest for path, digest in manifest.items()}
    if indexed_file_names is None:
        docstore = load_docstore()
        ref_doc_infos = docstore.get_all_ref_doc_info() or {}
        indexed_file_names = {
            str((getattr(info, "metadata", {}) or {}).get("file_name", "")).casefold()
            for info in ref_doc_infos.values()
        }
    documents = []

    if not root.exists():
        return documents

    resolved_root = root.resolve()
    for current_root, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(current_root) / filename
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
                continue

            resolved_path = path.resolve()
            try:
                resolved_path.relative_to(resolved_root)
            except ValueError:
                continue

            relative_id = path.relative_to(root).as_posix()
            repo_path = (Path(os.path.normpath(str(root))) / Path(relative_id)).as_posix()
            try:
                manifest_keys = (
                    normalize_path(repo_path),
                    normalize_path(Path(root.name) / Path(relative_id)),
                )
                indexed_hash = next(
                    (normalized_manifest[key] for key in manifest_keys if key in normalized_manifest),
                    None,
                )
                current_hash = file_hash(str(path))
                ready = indexed_hash == current_hash and filename.casefold() in indexed_file_names
                stat = path.stat()
            except OSError:
                continue

            documents.append({
                "document_id": relative_id,
                "filename": filename,
                "type": path.suffix[1:].upper(),
                "size_bytes": stat.st_size,
                "uploaded_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc),
                "status": "ready" if ready else "not_indexed",
            })

    return sorted(documents, key=lambda document: document["filename"].casefold())


def document_stats(documents: list[dict]) -> dict:
    ready_count = sum(document["status"] == "ready" for document in documents)
    return {
        "total_documents": len(documents),
        "ready_documents": ready_count,
        "not_indexed_documents": len(documents) - ready_count,
        "total_size_bytes": sum(document["size_bytes"] for document in documents),
    }


def delete_document(
    document_id: str,
    docstore,
    vector_store,
    data_dir: str | Path = DATA_DIR,
    manifest_path: str | Path = MANIFEST_PATH,
    storage_dir: str | Path = STORAGE_DIR,
) -> None:
    target = resolve_document_path(document_id, data_dir)
    if not target.is_file():
        raise FileNotFoundError("Document not found.")

    relative_id = PurePosixPath(document_id.replace("\\", "/"))
    manifest_match_keys = {
        normalize_path(Path(os.path.normpath(str(data_dir))) / Path(relative_id)),
        normalize_path(Path(Path(data_dir).name) / Path(relative_id)),
    }
    filename = target.name.casefold()
    ref_doc_infos = docstore.get_all_ref_doc_info() or {}
    matching_ref_ids = []

    for ref_doc_id, ref_info in ref_doc_infos.items():
        metadata = getattr(ref_info, "metadata", {}) or {}
        source_path = metadata.get("file_path")
        source_name = metadata.get("file_name", "")
        if source_path and normalize_path(source_path) in manifest_match_keys:
            matching_ref_ids.append(ref_doc_id)
        elif not source_path and str(source_name).casefold() == filename:
            matching_ref_ids.append(ref_doc_id)

    for ref_doc_id in matching_ref_ids:
        vector_store.delete(ref_doc_id=ref_doc_id)
        docstore.delete_ref_doc(ref_doc_id, raise_error=False)

    docstore.persist(persist_path=str(Path(storage_dir) / "docstore.json"))

    manifest = _read_manifest(manifest_path)
    for manifest_path_key in list(manifest):
        if normalize_path(manifest_path_key) in manifest_match_keys:
            del manifest[manifest_path_key]
    _write_manifest(manifest_path, manifest)

    target.unlink()


def _read_manifest(manifest_path: str | Path) -> dict:
    import json

    path = Path(manifest_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def _write_manifest(manifest_path: str | Path, manifest: dict) -> None:
    import json

    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

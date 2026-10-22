import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.documents import delete_document, document_stats, list_documents, resolve_document_path


class FakeDocstore:
    def __init__(self, refs):
        self.refs = refs
        self.deleted_ref_ids = []

    def get_all_ref_doc_info(self):
        return self.refs.copy()

    def delete_ref_doc(self, ref_doc_id, raise_error=False):
        self.deleted_ref_ids.append(ref_doc_id)
        self.refs.pop(ref_doc_id, None)

    def persist(self, persist_path):
        self.persist_path = persist_path


class FakeVectorStore:
    def __init__(self):
        self.deleted_ref_ids = []

    def delete(self, ref_doc_id):
        self.deleted_ref_ids.append(ref_doc_id)


class DocumentServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base_dir = Path(self.temp_dir.name)
        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir()
        self.manifest_path = self.base_dir / "manifest.json"
        self.storage_dir = self.base_dir / "storage"

    def write_manifest(self, entries):
        self.manifest_path.write_text(json.dumps(entries), encoding="utf-8")

    def test_listing_and_statistics_use_files_and_manifest_hashes(self):
        ready_content = b"published campus source"
        (self.data_dir / "guide.txt").write_bytes(ready_content)
        (self.data_dir / "draft.pdf").write_bytes(b"pending")
        self.write_manifest({
            "data/guide.txt": hashlib.sha256(ready_content).hexdigest(),
        })

        documents = list_documents(
            self.data_dir,
            self.manifest_path,
            indexed_file_names={"guide.txt"},
        )

        self.assertEqual([item["status"] for item in documents], ["not_indexed", "ready"])
        self.assertEqual(document_stats(documents), {
            "total_documents": 2,
            "ready_documents": 1,
            "not_indexed_documents": 1,
            "total_size_bytes": len(ready_content) + len(b"pending"),
        })

    def test_delete_removes_file_manifest_docstore_and_vector_refs(self):
        content = b"published campus source"
        file_path = self.data_dir / "guide.txt"
        file_path.write_bytes(content)
        self.write_manifest({
            "data/guide.txt": hashlib.sha256(content).hexdigest(),
            "data/keep.txt": "unrelated-hash",
        })
        docstore = FakeDocstore({
            "source-document-id": SimpleNamespace(
                metadata={"file_path": "data/guide.txt", "file_name": "guide.txt"},
            ),
        })
        vector_store = FakeVectorStore()

        delete_document(
            "guide.txt",
            docstore,
            vector_store,
            self.data_dir,
            self.manifest_path,
            self.storage_dir,
        )

        self.assertFalse(file_path.exists())
        self.assertEqual(docstore.deleted_ref_ids, ["source-document-id"])
        self.assertEqual(vector_store.deleted_ref_ids, ["source-document-id"])
        self.assertEqual(json.loads(self.manifest_path.read_text(encoding="utf-8")), {
            "data/keep.txt": "unrelated-hash",
        })
        self.assertEqual(docstore.persist_path, str(self.storage_dir / "docstore.json"))

    def test_document_id_cannot_escape_knowledge_directory(self):
        with self.assertRaises(ValueError):
            resolve_document_path("../outside.txt", self.data_dir)

    def test_manifest_hash_without_docstore_source_is_not_ready(self):
        content = b"published campus source"
        (self.data_dir / "guide.txt").write_bytes(content)
        self.write_manifest({
            "data/guide.txt": hashlib.sha256(content).hexdigest(),
        })

        document = list_documents(
            self.data_dir,
            self.manifest_path,
            indexed_file_names=set(),
        )[0]

        self.assertEqual(document["status"], "not_indexed")

    def test_empty_text_document_is_rejected_from_ingestion(self):
        from app.ingest import load_documents_manually

        empty_text = self.data_dir / "empty.txt"
        empty_text.write_text("  \n\t", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "no extractable text"):
            load_documents_manually([str(empty_text)])


if __name__ == "__main__":
    unittest.main()

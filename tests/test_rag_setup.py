import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import rag_setup


class HybridRetrieverTests(unittest.TestCase):
    def test_missing_local_docstore_falls_back_to_vector_retriever(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(rag_setup, "STORAGE_DIR", temp_dir),
                patch("app.rag_setup.get_vector_store", return_value=object()),
                patch("app.rag_setup.VectorStoreIndex.from_vector_store") as from_vector_store,
                patch("app.rag_setup.BM25Retriever.from_defaults") as make_bm25_retriever,
            ):
                vector_index = from_vector_store.return_value
                vector_retriever = vector_index.as_retriever.return_value

                retriever = rag_setup.get_hybrid_retriever(similarity_top_k=5)

        self.assertIs(retriever, vector_retriever)
        make_bm25_retriever.assert_not_called()

    def test_existing_local_docstore_keeps_hybrid_retrieval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "docstore.json").touch()
            with (
                patch.object(rag_setup, "STORAGE_DIR", temp_dir),
                patch("app.rag_setup.get_vector_store", return_value=object()),
                patch("app.rag_setup.VectorStoreIndex.from_vector_store") as from_vector_store,
                patch("app.rag_setup.SimpleDocumentStore.from_persist_dir") as load_docstore,
                patch("app.rag_setup.BM25Retriever.from_defaults") as make_bm25_retriever,
                patch("app.rag_setup.QueryFusionRetriever") as make_fusion_retriever,
            ):
                vector_retriever = from_vector_store.return_value.as_retriever.return_value
                bm25_retriever = make_bm25_retriever.return_value
                fusion_retriever = make_fusion_retriever.return_value

                retriever = rag_setup.get_hybrid_retriever(similarity_top_k=5)

        self.assertIs(retriever, fusion_retriever)
        make_bm25_retriever.assert_called_once_with(
            docstore=load_docstore.return_value,
            similarity_top_k=5,
        )
        make_fusion_retriever.assert_called_once()
        self.assertEqual(
            make_fusion_retriever.call_args.kwargs["retrievers"],
            [vector_retriever, bm25_retriever],
        )


if __name__ == "__main__":
    unittest.main()

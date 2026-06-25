"""
week5/tests/test_chromadb.py
============================
Unit tests for the ChromaDB Manager module.
Verifies CRUD operations on collections, document insertions, metadata filters, and mock fallback modes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.rag.vector_db.chromadb_manager import ChromaDbManager


class TestChromaDbManager:
    """Validate collection indexing, document updates, and search query scoring."""

    def test_chroma_mock_initialization(self):
        """Verify manager initializes correctly in forced mock mode."""
        manager = ChromaDbManager(force_mock=True)
        assert manager.is_mock_mode is True

    @pytest.mark.parametrize("force_mock", [True, False])
    def test_collections_crud_and_query_cycle(self, force_mock):
        """Verify standard CRUD cycle on styles, trends, brands, recs, and user preferences."""
        manager = ChromaDbManager(force_mock=force_mock)

        
        # Test collections for styles, trends, brands, recommendations, and preferences
        collections_to_test = [
            "fashion_styles",
            "trends",
            "brand_knowledge",
            "recommendations",
            "user_preferences"
        ]

        for col_name in collections_to_test:
            # 1. Create collection
            collection = manager.create_collection(col_name)
            assert collection is not None
            
            # 2. Insert documents
            ids = ["item_1", "item_2"]
            docs = ["Black oversized streetwear hoodie", "Red velvet luxury gown"]
            metas = [
                {"category": col_name, "color": "black", "style": "streetwear"},
                {"category": col_name, "color": "red", "style": "luxury"}
            ]
            embeddings = np.random.randn(2, 384).astype(np.float32)

            manager.insert_documents(
                collection_name=col_name,
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeddings
            )

            # 3. Query/Search documents with metadata filter
            search_res = manager.search_documents(
                collection_name=col_name,
                query_text="streetwear",
                n_results=1,
                where={"color": "black"}
            )
            assert len(search_res) == 1
            assert search_res[0]["id"] == "item_1"
            assert "streetwear" in search_res[0]["document"].lower()
            
            # Query by vector
            query_emb = np.random.randn(384).astype(np.float32)
            search_res_vec = manager.search_documents(
                collection_name=col_name,
                query_embeddings=query_emb,
                n_results=2
            )
            assert len(search_res_vec) == 2

            # 4. Update documents
            new_docs = ["Updated black oversized streetwear hoodie", "Red velvet luxury gown"]
            new_metas = [
                {"category": col_name, "color": "black", "style": "streetwear", "fit": "oversized"},
                {"category": col_name, "color": "red", "style": "luxury"}
            ]
            manager.update_documents(
                collection_name=col_name,
                ids=ids,
                documents=new_docs,
                metadatas=new_metas
            )
            
            search_res_updated = manager.search_documents(
                collection_name=col_name,
                query_text="hoodie",
                n_results=1,
                where={"fit": "oversized"}
            )
            assert len(search_res_updated) == 1
            assert search_res_updated[0]["id"] == "item_1"

            # 5. Delete documents
            manager.delete_documents(collection_name=col_name, ids=["item_1"])
            search_res_deleted = manager.search_documents(
                collection_name=col_name,
                query_text="hoodie",
                n_results=2
            )
            assert not any(item["id"] == "item_1" for item in search_res_deleted)

    def test_chroma_exception_coverage(self):
        """Verify error handling routes correctly when database operations fail."""
        import unittest.mock as mock
        
        # Trigger directory creation exception to test fallback block
        with mock.patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            manager = ChromaDbManager(persist_directory="test_path", force_mock=False)
            assert manager.is_mock_mode is True

    def test_chroma_persistent_client(self):
        """Verify persistent client connects and creates files."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            manager = ChromaDbManager(persist_directory=tmpdir, force_mock=False)
            assert manager.client is not None





"""
week5/tests/test_fashion_rag_pipeline.py
========================================
Unit tests for the Fashion RAG Pipeline module.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.rag.fashion_rag_pipeline import FashionRAGPipeline
from src.rag.retrieval.fashion_retriever import FashionRetriever
from src.rag.vector_db.chromadb_manager import ChromaDbManager


@pytest.fixture
def mock_embedder():
    """EmbeddingsGenerator in mock mode."""
    return EmbeddingsGenerator(force_mock=True)


@pytest.fixture
def mock_db_manager():
    """ChromaDbManager in mock mode."""
    return ChromaDbManager(force_mock=True)


@pytest.fixture
def mock_retriever(mock_embedder, mock_db_manager):
    """FashionRetriever initialized with mock engines."""
    return FashionRetriever(embedder=mock_embedder, db_manager=mock_db_manager)


class TestFashionRAGPipeline:
    """Validate full pipeline execution, citation accuracy, source tracking, and confidence scoring."""

    def test_pipeline_initialization(self, mock_retriever, mock_embedder):
        """Verify the pipeline initializes correctly."""
        pipeline = FashionRAGPipeline(retriever=mock_retriever, embedder=mock_embedder)
        assert pipeline.retriever == mock_retriever
        assert pipeline.embedder == mock_embedder

    def test_assemble_context(self, mock_retriever, mock_embedder):
        """Verify assembly of retrieved items into context blocks."""
        pipeline = FashionRAGPipeline(retriever=mock_retriever, embedder=mock_embedder)

        # 1. Empty context
        assert pipeline.assemble_context([]) == "No relevant context found."

        # 2. Structured items context
        items = [
            {"id": "doc_1", "document": "Wide leg pants are trending.", "metadata": {"source": "trends.json"}},
            {"id": "doc_2", "document": "Puffer jackets for winter.", "metadata": {"source": "winter.txt"}}
        ]
        context = pipeline.assemble_context(items)
        assert "Document [1]" in context
        assert "ID: doc_1" in context
        assert "Source: trends.json" in context
        assert "Content: Wide leg pants are trending." in context
        assert "Document [2]" in context
        assert "ID: doc_2" in context
        assert "Source: winter.txt" in context
        assert "Content: Puffer jackets for winter." in context

    def test_calculate_confidence(self, mock_retriever, mock_embedder):
        """Verify confidence scoring maps correctly to distance ranges."""
        pipeline = FashionRAGPipeline(retriever=mock_retriever, embedder=mock_embedder)

        # 1. Empty items
        assert pipeline.calculate_confidence([]) == 0.0

        # 2. Perfect distance (0.0 distance -> 1.0 similarity)
        items_perfect = [{"distance": 0.0}]
        conf_perfect = pipeline.calculate_confidence(items_perfect)
        # Similarity = 1.0, count bonus = 0.8 + 0.05 * 1 = 0.85 -> conf = 0.85
        assert abs(conf_perfect - 0.85) < 1e-5

        # 3. Poor similarity (0.9 distance -> 0.1 similarity)
        items_poor = [{"distance": 0.9}]
        conf_poor = pipeline.calculate_confidence(items_poor)
        # Similarity = 0.1, count bonus = 0.85 -> conf = 0.085
        assert abs(conf_poor - 0.085) < 1e-5

        # 4. Multiple matches bonus
        items_multi = [{"distance": 0.2}, {"distance": 0.2}, {"distance": 0.2}, {"distance": 0.2}]
        # Similarity = 0.8
        # count bonus = 0.8 + 0.05 * 4 = 1.0 -> conf = 0.8 * 1.0 = 0.8
        conf_multi = pipeline.calculate_confidence(items_multi)
        assert abs(conf_multi - 0.8) < 1e-5

    def test_generate_response(self, mock_retriever, mock_embedder):
        """Verify citation grounding and source tracking in response generation."""
        pipeline = FashionRAGPipeline(retriever=mock_retriever, embedder=mock_embedder)

        # 1. Empty fallback
        empty_gen = pipeline.generate_response("query", "context", [])
        assert "could not find any specific domain knowledge" in empty_gen["response"]
        assert empty_gen["citations"] == []
        assert empty_gen["sources"] == []

        # 2. Grounded statements with overlap and default top-1
        items = [
            {
                "id": "doc_hoodie",
                "document": "Streetwear oversized hoodie with graphics. Made from fleece.",
                "metadata": {"source": "streetwear_dataset.csv"}
            },
            {
                "id": "doc_gown",
                "document": "Luxury silk gowns in red color.",
                "metadata": {"source": "luxury_catalog.json"}
            }
        ]

        # Query matches hoodie, first doc matches overlap, second does not.
        # But top-1 (first document) always gets generated as fallback to ensure at least one grounded sentence.
        res_hoodie = pipeline.generate_response("oversized hoodie", "context", items)
        assert "doc_hoodie" in res_hoodie["citations"]
        assert "streetwear_dataset.csv" in res_hoodie["sources"]
        assert "streetwear oversized hoodie with graphics" in res_hoodie["response"].lower()

        # Query matches gown
        res_gown = pipeline.generate_response("silk gowns", "context", items)
        assert "doc_gown" in res_gown["citations"]
        assert "luxury_catalog.json" in res_gown["sources"]
        assert "luxury silk gowns in red color" in res_gown["response"].lower()

    def test_run_pipeline(self, mock_retriever, mock_embedder, mock_db_manager):
        """Verify full run_pipeline coordinator outputs correct schema payload."""
        pipeline = FashionRAGPipeline(retriever=mock_retriever, embedder=mock_embedder)
        col_name = "pipeline_test_col"

        # Populate database
        docs = ["Activewear leggings with pocket.", "Cashmere trench coat for luxury look."]
        embeddings = mock_embedder.embed_batch(docs)
        mock_db_manager.insert_documents(
            collection_name=col_name,
            ids=["doc_1", "doc_2"],
            documents=docs,
            metadatas=[{"source": "active.txt", "popularity": "high"}, {"source": "luxury.txt", "popularity": "medium"}],
            embeddings=embeddings
        )

        # Run pipeline
        results = pipeline.run_pipeline(
            query="luxury cashmere coat",
            collection_name=col_name,
            n_results=2,
            search_type="semantic"
        )

        assert results["query"] == "luxury cashmere coat"
        assert len(results["response"]) > 0
        assert len(results["citations"]) > 0
        assert "doc_2" in results["citations"]
        assert "luxury.txt" in results["sources"]
        assert 0.0 <= results["confidence_score"] <= 1.0
        assert len(results["ranked_results"]) == 2
        assert "context_assembled" in results

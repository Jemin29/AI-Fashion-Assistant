"""
week5/rag/fashion_rag_pipeline.py
=================================
Fashion RAG Pipeline Coordinator.
Implements the full end-to-end Retrieval-Augmented Generation loop:
User Query -> Embedding -> Vector Search -> Context Retrieval -> Response Generation.
Includes context assembly, inline citations, source tracking, and confidence scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.rag.retrieval.fashion_retriever import FashionRetriever


class FashionRAGPipeline:
    """
    Coordinates semantic vector search, context aggregation, source tracking,
    confidence scoring, and citation-grounded response generation.
    """

    def __init__(
        self,
        retriever: FashionRetriever,
        embedder: EmbeddingsGenerator
    ) -> None:
        """
        Initialize the Fashion RAG Pipeline.

        Parameters
        ----------
        retriever : FashionRetriever
            The retrieval engine wrapping ChromaDB.
        embedder : EmbeddingsGenerator
            Dense embeddings generation engine.
        """
        self.retriever = retriever
        self.embedder = embedder
        logger.info("FashionRAGPipeline successfully initialized.")

    def assemble_context(self, retrieved_items: List[Dict[str, Any]]) -> str:
        """
        Assemble retrieved documents into a formatted context block.

        Parameters
        ----------
        retrieved_items : List[Dict[str, Any]]
            Search results from the retriever.

        Returns
        -------
        str
            Aggregated, structured context string block.
        """
        if not retrieved_items:
            return "No relevant context found."

        context_blocks = []
        for idx, item in enumerate(retrieved_items, 1):
            item_id = item.get("id", f"item_{idx}")
            doc_text = item.get("document", "").strip()
            metadata = item.get("metadata", {})
            source = metadata.get("source", "unknown_source")

            block = (
                f"Document [{idx}] | ID: {item_id} | Source: {source}\n"
                f"Content: {doc_text}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def calculate_confidence(self, retrieved_items: List[Dict[str, Any]]) -> float:
        """
        Calculate the pipeline's response confidence score based on similarity values.

        Parameters
        ----------
        retrieved_items : List[Dict[str, Any]]
            List of matching records with distances.

        Returns
        -------
        float
            Confidence score between 0.0 and 1.0.
        """
        if not retrieved_items:
            logger.warning("No retrieved items for confidence scoring. Confidence: 0.0")
            return 0.0

        similarities = []
        for item in retrieved_items:
            distance = item.get("distance", 0.0)
            # Map Chroma distance to a similarity metric in range [0, 1]
            similarity = 1.0 - max(0.0, min(1.0, distance))
            similarities.append(similarity)

        # Compute average vector similarity
        avg_similarity = sum(similarities) / len(similarities)

        # Scale confidence based on the number of relevant documents retrieved
        # Retrieving multiple relevant matches increases confidence slightly
        count_bonus = min(1.0, 0.8 + 0.05 * len(retrieved_items))
        confidence = avg_similarity * count_bonus

        return float(max(0.0, min(1.0, confidence)))

    def generate_response(
        self,
        query: str,
        context: str,
        retrieved_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize a grounded design response citing source documents.

        Parameters
        ----------
        query : str
            Input user request.
        context : str
            Formatted context block.
        retrieved_items : List[Dict[str, Any]]
            List of matching documents.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing generated response text, citations used, and source files.
        """
        if not retrieved_items:
            return {
                "response": "I could not find any specific domain knowledge to answer your request.",
                "citations": [],
                "sources": []
            }

        response_parts = []
        citations_used = []
        sources_used = {}

        query_words = set(query.lower().split())

        # Ground response statements on retrieved documents containing keyword overlap or high relevance
        for item in retrieved_items:
            item_id = item["id"]
            doc_text = item["document"]
            metadata = item.get("metadata", {})
            source = metadata.get("source", "unknown_source")

            doc_words = set(doc_text.lower().split())
            overlap = query_words & doc_words

            # Generate statement if there's semantic overlap or it's the most similar document
            if overlap or len(response_parts) == 0:
                # Extract first sentence for grounded facts representation
                sentence = doc_text.split(".")[0].strip()
                if not sentence.endswith("."):
                    sentence += "."

                statement = f"According to fashion records, {sentence.lower()} [{item_id}]"
                response_parts.append(statement)
                citations_used.append(item_id)
                sources_used[item_id] = source

        response_text = " ".join(response_parts)

        return {
            "response": response_text,
            "citations": citations_used,
            "sources": list(set(sources_used.values()))
        }

    def run_pipeline(
        self,
        query: str,
        collection_name: str = "fashion_styles",
        n_results: int = 3,
        search_type: str = "semantic",
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete Retrieval-Augmented Generation pipeline cycle.

        Parameters
        ----------
        query : str
        collection_name : str
        n_results : int
        search_type : str
        where : dict, optional

        Returns
        -------
        Dict[str, Any]
            Structured generation results payload.
        """
        logger.info(f"Running Fashion RAG Pipeline for query: '{query}'...")

        # 1. Vector Search
        retrieved_items = self.retriever.retrieve(
            query=query,
            search_type=search_type,
            collection_name=collection_name,
            n_results=n_results,
            where=where
        )

        # 2. Context Assembly
        context = self.assemble_context(retrieved_items)

        # 3. Confidence Scoring
        confidence = self.calculate_confidence(retrieved_items)

        # 4. Reranking / Metadata weighting
        ranked_results = self.retriever.rank_results(retrieved_items)

        # 5. Response Generation
        generation = self.generate_response(query, context, ranked_results)

        return {
            "query": query,
            "response": generation["response"],
            "context_assembled": context,
            "confidence_score": confidence,
            "citations": generation["citations"],
            "sources": generation["sources"],
            "ranked_results": ranked_results
        }

"""
week5/retrieval/fashion_retriever.py
===================================
Fashion Retrieval Engine.
Interfaces with ChromaDB and EmbeddingsGenerator to support semantic searches,
similarity searches, trend/style/brand lookups, and metadata-aware reranking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger

from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.rag.vector_db.chromadb_manager import ChromaDbManager


class FashionRetriever:
    """
    Retrieves and ranks fashion documents, styles, trends, and brand profiles
    using semantic vector search, document similarity, and metadata-blended ranking.
    """

    def __init__(
        self,
        embedder: EmbeddingsGenerator,
        db_manager: ChromaDbManager
    ) -> None:
        """
        Initialize the Fashion Retriever.

        Parameters
        ----------
        embedder : EmbeddingsGenerator
            Dense vector embeddings generator.
        db_manager : ChromaDbManager
            ChromaDB collections interface.
        """
        self.embedder = embedder
        self.db_manager = db_manager
        logger.info("FashionRetriever initialized successfully.")

    def retrieve(
        self,
        query: str,
        search_type: str = "semantic",
        collection_name: Optional[str] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve fashion records using semantic vector search.

        Parameters
        ----------
        query : str
            Search query string.
        search_type : str
            Type of search mode: "semantic", "trend", "style", or "brand".
        collection_name : str, optional
            Direct collection override. If omitted, maps automatically based on search_type.
        n_results : int
            Number of matching documents to return.
        where : dict, optional
            Metadata filter queries.

        Returns
        -------
        List[Dict[str, Any]]
            List of parsed matching documents with IDs, text, metadata, and distance metrics.
        """
        if not query:
            logger.warning("Empty query passed to retrieve(). Returning empty list.")
            return []

        # Resolve collection name based on search type if not explicitly provided
        if not collection_name:
            search_clean = search_type.lower().strip()
            if search_clean == "trend":
                collection_name = "trends"
            elif search_clean == "style":
                collection_name = "fashion_styles"
            elif search_clean == "brand":
                collection_name = "brand_knowledge"
            else:
                collection_name = "fashion_styles"

        logger.info(f"Retrieving matching items from '{collection_name}' | Type: {search_type} | Query: '{query}'")

        # Generate dense vector representation for the query
        query_vector = self.embedder.embed_text(query)

        # Query the database
        results = self.db_manager.search_documents(
            collection_name=collection_name,
            query_embeddings=query_vector,
            n_results=n_results,
            where=where
        )

        return results

    def search_similar(
        self,
        item_id: str,
        collection_name: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find documents similar to a target document ID within a specific collection.

        Parameters
        ----------
        item_id : str
            The ID of the document to find matches for.
        collection_name : str
            The collection name containing the document.
        n_results : int
            Number of similar items to return.

        Returns
        -------
        List[Dict[str, Any]]
            List of top similar documents, excluding the query document itself.
        """
        logger.info(f"Searching for items similar to '{item_id}' in collection '{collection_name}'...")

        collection = self.db_manager.create_collection(collection_name)

        doc_text = ""
        doc_emb = None

        # Check if the collection is a Mock Collection or Native ChromaDB collection
        if hasattr(collection, "ids"):  # MockChromaCollection fallback path
            if item_id in collection.ids:
                pos = collection.ids.index(item_id)
                doc_text = collection.documents[pos]
                doc_emb = collection.embeddings[pos]
            else:
                logger.warning(f"Item ID '{item_id}' not found in mock collection '{collection_name}'.")
                return []
        else:  # Native ChromaDB Collection path
            try:
                # Include embeddings to query vector-to-vector directly
                doc_res = collection.get(ids=[item_id], include=["documents", "embeddings"])
                if doc_res and doc_res["ids"]:
                    doc_text = doc_res["documents"][0]
                    if doc_res.get("embeddings") is not None and len(doc_res["embeddings"]) > 0:
                        doc_emb = doc_res["embeddings"][0]
                else:
                    logger.warning(f"Item ID '{item_id}' not found in ChromaDB collection '{collection_name}'.")
                    return []
            except Exception as err:
                logger.error(f"Error fetching item '{item_id}' from ChromaDB: {err}")
                return []

        # Perform similarity search using embedding (preferred) or document text
        if doc_emb is not None:
            # Convert embedding to numpy array format if it is a list
            if isinstance(doc_emb, list):
                q_emb = np.array(doc_emb, dtype=np.float32)
            else:
                q_emb = doc_emb
            
            raw_results = self.db_manager.search_documents(
                collection_name=collection_name,
                query_embeddings=q_emb,
                n_results=n_results + 1  # Get extra result to account for self-exclusion
            )
        else:
            raw_results = self.db_manager.search_documents(
                collection_name=collection_name,
                query_text=doc_text,
                n_results=n_results + 1
            )

        # Filter out the query document itself and limit to requested count
        filtered_results = [r for r in raw_results if r["id"] != item_id][:n_results]
        return filtered_results

    def rank_results(
        self,
        results: List[Dict[str, Any]],
        scoring_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank search results using a blended score of vector similarity and metadata metrics.

        Parameters
        ----------
        results : List[Dict[str, Any]]
            Search results returned by retrieve() or search_similar().
        scoring_weights : Dict[str, float], optional
            Weights for ranking components: 'vector_similarity', 'popularity', 'growth_rate'.
            Defaults to 80% Vector Similarity, 20% Popularity, 0% Growth.

        Returns
        -------
        List[Dict[str, Any]]
            Reranked results list sorted descending by blended scores.
        """
        if not results:
            return []

        if scoring_weights is None:
            scoring_weights = {
                "vector_similarity": 0.8,
                "popularity": 0.2,
                "growth_rate": 0.0
            }

        # Normalize weights so they sum to 1.0
        total_weight = sum(scoring_weights.values())
        if total_weight > 0.0:
            weights = {k: v / total_weight for k, v in scoring_weights.items()}
        else:
            weights = {"vector_similarity": 1.0, "popularity": 0.0, "growth_rate": 0.0}

        ranked_results = []

        for item in results:
            # Map distance (Chroma L2 or cosine distance) to similarity [0, 1]
            distance = item.get("distance", 0.0)
            vector_sim = 1.0 - max(0.0, min(1.0, distance))

            metadata = item.get("metadata", {})

            # 1. Parse Popularity Score
            pop_score = 0.5  # Default fallback
            # Support popularity_score and popularity fields
            pop_val = metadata.get("popularity_score", metadata.get("popularity", 0.5))
            if isinstance(pop_val, (int, float)):
                pop_score = float(pop_val)
            elif isinstance(pop_val, str):
                pop_map = {"high": 1.0, "medium": 0.5, "low": 0.1}
                pop_score = pop_map.get(pop_val.lower().strip(), 0.5)

            # 2. Parse Growth Rate
            growth_rate = 0.0  # Default fallback
            growth_val = metadata.get("growth_rate", 0.0)
            try:
                growth_rate = float(growth_val)
            except (ValueError, TypeError):
                pass
            # Map growth rate (e.g. [-1.0, 1.0]) to [0.0, 1.0] for blending
            norm_growth = max(0.0, min(1.0, (growth_rate + 1.0) / 2.0))

            # Compute blended score
            blended_score = (
                weights.get("vector_similarity", 0.0) * vector_sim +
                weights.get("popularity", 0.0) * pop_score +
                weights.get("growth_rate", 0.0) * norm_growth
            )

            # Construct ranked dictionary
            ranked_item = {
                **item,
                "vector_similarity": vector_sim,
                "popularity_score": pop_score,
                "growth_rate_normalized": norm_growth,
                "blended_score": blended_score
            }
            ranked_results.append(ranked_item)

        # Sort descending by blended score
        ranked_results.sort(key=lambda x: x["blended_score"], reverse=True)
        return ranked_results

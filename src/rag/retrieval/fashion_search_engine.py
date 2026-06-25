"""
week5/retrieval/fashion_search_engine.py
========================================
Fashion Search Engine.
Provides unified keyword queries, semantic searches, and metadata filters
across fashion design style, brand knowledge, and trend datasets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from src.rag.retrieval.fashion_retriever import FashionRetriever
from src.rag.vector_db.chromadb_manager import ChromaDbManager


class FashionSearchEngine:
    """
    Unified retrieval portal supporting keyword searching, semantic vector querying,
    dynamic metadata-blended ranking, metadata filtering, and custom attribute sorting.
    """

    def __init__(
        self,
        retriever: FashionRetriever,
        db_manager: ChromaDbManager
    ) -> None:
        """
        Initialize the Fashion Search Engine.

        Parameters
        ----------
        retriever : FashionRetriever
            Semantic retrieval coordinator.
        db_manager : ChromaDbManager
            ChromaDB database collection manager.
        """
        self.retriever = retriever
        self.db_manager = db_manager
        logger.info("FashionSearchEngine successfully initialized.")

    def search(
        self,
        query: str,
        search_type: str = "semantic",
        collection_name: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        scoring_weights: Optional[Dict[str, float]] = None,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for fashion items using keyword, semantic, or categorized queries.

        Parameters
        ----------
        query : str
            Query string parameter.
        search_type : str
            Type of search: "keyword", "semantic", "style", "brand", or "trend".
        collection_name : str, optional
            ChromaDB collection override name.
        filter_metadata : Dict[str, Any], optional
            ChromaDB metadata filter query (where clause).
        scoring_weights : Dict[str, float], optional
            Weights for ranking component blend: 'vector_similarity', 'popularity', 'growth_rate'.
        sort_by : str, optional
            Field name to sort results by (e.g. 'popularity_score', 'blended_score', 'price').
        ascending : bool
            True to sort ascending, False for descending.
        limit : int
            Number of results to return.

        Returns
        -------
        List[Dict[str, Any]]
            Filtered, ranked, sorted, and limited query results.
        """
        logger.info(
            f"FashionSearchEngine.search | query='{query}' | type={search_type} | "
            f"collection={collection_name} | filters={filter_metadata} | sort={sort_by}"
        )

        search_clean = search_type.lower().strip()
        
        # 1. Resolve target collection name if not specified
        if not collection_name:
            if search_clean == "trend":
                collection_name = "trends"
            elif search_clean == "style":
                collection_name = "fashion_styles"
            elif search_clean == "brand":
                collection_name = "brand_knowledge"
            else:
                collection_name = "fashion_styles"

        results = []

        # 2. Execute keyword search or semantic search
        if search_clean == "keyword":
            # Keyword text search directly using ChromaDB query_text parameter
            results = self.db_manager.search_documents(
                collection_name=collection_name,
                query_text=query,
                n_results=limit * 3,
                where=filter_metadata
            )
        else:
            # Dense vector semantic retrieval
            results = self.retriever.retrieve(
                query=query,
                search_type=search_type,
                collection_name=collection_name,
                n_results=limit * 3,
                where=filter_metadata
            )

        if not results:
            return []

        # 3. Apply blended ranking using vector, popularity, and growth weights
        ranked_results = self.retriever.rank_results(results, scoring_weights)

        # 4. Apply custom sorting if requested
        if sort_by:
            def sort_key(item: Dict[str, Any]) -> Any:
                # First check if the key is at the root level of the item dict
                if sort_by in item:
                    val = item[sort_by]
                else:
                    # Otherwise, check in the metadata dict
                    metadata = item.get("metadata", {})
                    val = metadata.get(sort_by, None)

                # Return a default comparable value if the key doesn't exist
                if val is None:
                    return 0.0 if not isinstance(sort_by, str) else ""
                return val

            try:
                ranked_results.sort(key=sort_key, reverse=not ascending)
            except Exception as err:
                logger.warning(f"Error sorting search results by key '{sort_by}': {err}")

        # 5. Clamp output count to limit
        return ranked_results[:limit]

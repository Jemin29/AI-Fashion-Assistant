"""
week5/retrieval/hybrid_retriever.py
===================================
Hybrid Retriever for Week 5 RAG system.
Combines pure-Python TF-IDF keyword retrieval with FAISS dense vector retrieval
using normalized score blending.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

from src.utils.config_manager import RetrievalConfig, get_default_config
from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.data.knowledge_base.fashion_knowledge_base import FashionKnowledgeBase, KnowledgeItem
from src.data.knowledge_base.trend_dataset_builder import TrendDatasetBuilder, TrendItem
from src.rag.vector_db.vector_indexer import VectorIndexer


# =============================================================================
# ── Keyword Retriever (Pure-Python TF-IDF)
# =============================================================================

class KeywordRetriever:
    """
    Lightweight, dependency-free in-memory TF-IDF keyword search engine.
    Computes normalized matching relevance scores for queries against text documents.
    """

    def __init__(self) -> None:
        self.doc_ids: List[str] = []
        self.docs: Dict[str, str] = {}
        self.idf: Dict[str, float] = {}
        self.doc_tfs: Dict[str, Dict[str, float]] = {}

    def tokenise(self, text: str) -> List[str]:
        """Convert text into lowercase words of length 2 or more."""
        return re.findall(r"\b\w{2,}\b", text.lower())

    def fit(self, items: List[Union[KnowledgeItem, TrendItem]]) -> None:
        """
        Build vocabulary, TF, and IDF models from a list of items.

        Parameters
        ----------
        items : list of KnowledgeItem or TrendItem
        """
        self.doc_ids = []
        self.docs = {}
        self.doc_tfs = {}
        self.idf = {}

        if not items:
            return

        doc_words: Dict[str, List[str]] = {}
        for item in items:
            self.doc_ids.append(item.id)
            if isinstance(item, KnowledgeItem):
                tags_str = " ".join(item.tags)
                text = f"{item.name} {item.category} {item.content} {tags_str}"
            elif isinstance(item, TrendItem):
                text = f"{item.name} {item.category} {item.description}"
            else:
                text = str(item)
            self.docs[item.id] = text
            doc_words[item.id] = self.tokenise(text)

        # Count Document Frequency (DF)
        df: Dict[str, int] = {}
        N = len(items)
        for doc_id, words in doc_words.items():
            unique_words = set(words)
            for w in unique_words:
                df[w] = df.get(w, 0) + 1

        # Compute Inverse Document Frequency (IDF) - smoothed
        for w, count in df.items():
            self.idf[w] = math.log((N - count + 0.5) / (count + 0.5) + 1.0)

        # Compute Term Frequencies (TF) per document
        for doc_id, words in doc_words.items():
            if not words:
                self.doc_tfs[doc_id] = {}
                continue
            word_counts: Dict[str, int] = {}
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1
            
            # Normalize TF by document length to prevent long document bias
            total_words = len(words)
            self.doc_tfs[doc_id] = {w: count / total_words for w, count in word_counts.items()}

        logger.debug(f"Fitted KeywordRetriever corpus with {N} items and {len(df)} vocabulary words.")

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Calculate TF-IDF match scores, normalized to [0.0, 1.0].

        Parameters
        ----------
        query : str
        top_k : int

        Returns
        -------
        list of tuple of (doc_id, score)
        """
        q_words = self.tokenise(query)
        if not q_words or not self.docs:
            return []

        scores: Dict[str, float] = {}
        for doc_id, tfs in self.doc_tfs.items():
            score = 0.0
            for w in q_words:
                if w in tfs:
                    score += tfs[w] * self.idf.get(w, 0.0)
            if score > 0.0:
                scores[doc_id] = score

        if not scores:
            return []

        # Min-Max Normalization to scale matching scores within [0.0, 1.0]
        max_val = max(scores.values())
        min_val = min(scores.values())
        span = max_val - min_val

        norm_scores: Dict[str, float] = {}
        if span > 1e-9:
            for doc_id, val in scores.items():
                norm_scores[doc_id] = (val - min_val) / span
        else:
            for doc_id in scores.keys():
                norm_scores[doc_id] = 1.0

        sorted_res = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_res[:top_k]


# =============================================================================
# ── Hybrid Retriever
# =============================================================================

class HybridRetriever:
    """
    Routable Hybrid Search system bridging Keyword Search and Dense Vector Search.
    Performs score normalization and blending according to configured weights.
    """

    def __init__(
        self,
        kb: FashionKnowledgeBase,
        embedder: EmbeddingsGenerator,
        indexer: VectorIndexer,
        trend_builder: Optional[TrendDatasetBuilder] = None,
        config: Optional[RetrievalConfig] = None
    ) -> None:
        """
        Initialize the Hybrid Retriever.

        Parameters
        ----------
        kb : FashionKnowledgeBase
        embedder : EmbeddingsGenerator
        indexer : VectorIndexer
        trend_builder : TrendDatasetBuilder, optional
        config : RetrievalConfig, optional
        """
        self.kb = kb
        self.embedder = embedder
        self.indexer = indexer
        self.trend_builder = trend_builder

        if config is None:
            self.config = get_default_config().retrieval
        else:
            self.config = config

        self.kw_retriever = KeywordRetriever()
        self.fit_retrievers()

    def fit_retrievers(self) -> None:
        """Fetch all documents from sources, build TF-IDF and build vector index."""
        # 1. Gather all documents
        items: List[Union[KnowledgeItem, TrendItem]] = []
        items.extend(self.kb.list_items())
        if self.trend_builder:
            items.extend(self.trend_builder.list_trends())

        # 2. Fit TF-IDF model
        self.kw_retriever.fit(items)

        # 3. Compile FAISS Vector Index
        self.indexer.clear()
        if items:
            item_ids = [item.id for item in items]
            embeddings = self.embedder.embed_items(items)
            self.indexer.add_items(item_ids, embeddings)
            
        logger.info(f"Fitted HybridRetriever with {len(items)} items into keyword and vector indexes.")

    def _resolve_item(self, item_id: str) -> Optional[Union[KnowledgeItem, TrendItem]]:
        """Look up item by ID in local memory stores."""
        if item_id.startswith("kb_"):
            return self.kb.get_item(item_id)
        elif item_id.startswith("trend_"):
            return self.trend_builder.get_trend(item_id) if self.trend_builder else None
        else:
            # Fallback check
            kb_item = self.kb.get_item(item_id)
            if kb_item:
                return kb_item
            if self.trend_builder:
                return self.trend_builder.get_trend(item_id)
            return None

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Union[KnowledgeItem, TrendItem], float]]:
        """
        Perform hybrid search routing, normalizing and blending outputs.

        Parameters
        ----------
        query : str
        top_k : int

        Returns
        -------
        list of tuple of (item, combined_score)
        """
        if not query:
            return []

        # 1. Determine retrieval route
        use_hybrid = self.config.hybrid_search
        
        kw_results: Dict[str, float] = {}
        vec_results: Dict[str, float] = {}

        # ── Keyword Scorer ──
        if use_hybrid or (self.config.keyword_weight > 0.0):
            # Fetch keyword matches
            kw_raw = self.kw_retriever.search(query, top_k=max(top_k * 3, 50))
            kw_results = dict(kw_raw)

        # ── Vector Scorer ──
        if use_hybrid or (self.config.vector_weight > 0.0):
            # Embed query
            query_vector = self.embedder.embed_text(query)
            # Search FAISS index
            vec_raw = self.indexer.search(query_vector, top_k=max(top_k * 3, 50))
            
            # Normalize vector search scores to [0.0, 1.0]
            for doc_id, dist in vec_raw:
                if self.indexer.config.index_type.upper() == "INNERPRODUCT":
                    # Cosine/IP similarity normalisation
                    norm_score = max(0.0, min(1.0, dist))
                else:
                    # L2 distance normalization: 1.0 / (1.0 + d)
                    norm_score = 1.0 / (1.0 + dist)
                vec_results[doc_id] = norm_score

        # ── Route Blending ──
        all_ids = set(kw_results.keys()) | set(vec_results.keys())
        blended_scores: List[Tuple[str, float]] = []

        for doc_id in all_ids:
            kw_score = kw_results.get(doc_id, 0.0)
            vec_score = vec_results.get(doc_id, 0.0)

            if use_hybrid:
                score = (self.config.keyword_weight * kw_score) + (self.config.vector_weight * vec_score)
            elif self.config.keyword_weight > 0.0:
                score = kw_score
            else:
                score = vec_score

            blended_scores.append((doc_id, score))

        # Sort combined results descending
        blended_scores.sort(key=lambda x: x[1], reverse=True)

        # Resolve items and yield top_k
        final_results = []
        for doc_id, score in blended_scores:
            item = self._resolve_item(doc_id)
            if item:
                final_results.append((item, score))
            if len(final_results) >= top_k:
                break

        return final_results

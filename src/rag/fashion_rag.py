"""
week5/rag/fashion_rag.py
========================
Fashion RAG Coordinator for Week 5.
Bridges retrieval, trend analysis, recommendations, and context augmentation
to generate final design responses.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from src.utils.config_manager import Week5Config, get_default_config
from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.data.knowledge_base.fashion_knowledge_base import FashionKnowledgeBase, KnowledgeItem
from src.data.knowledge_base.trend_dataset_builder import TrendDatasetBuilder, TrendItem
from src.recommendations.recommendation_engine import RecommendationEngine
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.trends.trend_analyzer import TrendAnalyzer
from src.rag.vector_db.vector_indexer import VectorIndexer



class FashionRAG:
    """
    Coordinator class implementing the complete Retrieval-Augmented Generation loop.
    Extracts semantic details, injects active trends, and synthesizes structured prompt contexts.
    """

    def __init__(
        self,
        config: Optional[Week5Config] = None,
        kb_path: Optional[str] = None,
        trend_db_path: Optional[str] = None,
        force_mock_embeddings: bool = False
    ) -> None:
        """
        Initialize the Fashion RAG Coordinator.

        Parameters
        ----------
        config : Week5Config, optional
        kb_path : str, optional
            Custom path to knowledge base JSON database.
        trend_db_path : str, optional
            Custom path to trend dataset JSON database.
        force_mock_embeddings : bool
            Force embedder to run in mock mode.
        """
        if config is None:
            self.config = get_default_config()
        else:
            self.config = config

        # 1. Initialize databases
        self.kb = FashionKnowledgeBase(db_path=kb_path)
        self.trend_dataset = TrendDatasetBuilder(db_path=trend_db_path)

        # 2. Initialize ML models and indexes
        self.embedder = EmbeddingsGenerator(
            config=self.config.embeddings,
            force_mock=force_mock_embeddings
        )
        self.indexer = VectorIndexer(
            config=self.config.vector_db,
            dimension=self.config.embeddings.dimension
        )

        # 3. Initialize search and analytical components
        self.retriever = HybridRetriever(
            kb=self.kb,
            embedder=self.embedder,
            indexer=self.indexer,
            trend_builder=self.trend_dataset,
            config=self.config.retrieval
        )
        
        self.trend_analyzer = TrendAnalyzer(config=self.config.trends)
        # Ingest default dataset trends into analyzer mentions to seed analyzer
        for trend in self.trend_dataset.list_trends():
            self.trend_analyzer.add_mention(trend.name)
            # Add metadata items as additional mentions for richness
            for val in trend.metadata.values():
                if isinstance(val, list):
                    for v in val:
                        self.trend_analyzer.add_mention(str(v))
                elif isinstance(val, str):
                    self.trend_analyzer.add_mention(val)


        self.recommendation_engine = RecommendationEngine(
            retriever=self.retriever,
            embedder=self.embedder,
            config=self.config.recommendations
        )

        logger.success("Fashion RAG Pipeline Coordinator fully initialized.")

    def augment_prompt(
        self,
        prompt: str,
        context_items: List[Union[KnowledgeItem, TrendItem]]
    ) -> str:
        """
        Construct a context-augmented prompt header containing retrieved knowledge.

        Parameters
        ----------
        prompt : str
            Raw user design prompt.
        context_items : list
            Retrieved items to format into context.

        Returns
        -------
        str
        """
        if not context_items:
            return f"No context retrieved.\n\nUser Query: {prompt}"

        context_blocks = []
        for i, item in enumerate(context_items, 1):
            if isinstance(item, KnowledgeItem):
                tags_str = ", ".join(item.tags)
                block = (
                    f"[{i}] Item ID: {item.id}\n"
                    f"    Category: {item.category} | Name: {item.name}\n"
                    f"    Relevance Details: {item.content}\n"
                    f"    Attributes: {tags_str}\n"
                )
            elif isinstance(item, TrendItem):
                block = (
                    f"[{i}] Trend ID: {item.id}\n"
                    f"    Category: {item.category} | Trend Name: {item.name}\n"
                    f"    Popularity: {item.popularity_score} | Description: {item.description}\n"
                )
            else:
                block = f"[{i}] Context: {str(item)}\n"
            context_blocks.append(block)

        context_section = "\n".join(context_blocks)

        augmented = (
            "=========================================================================\n"
            "FASHION RETRIEVAL CONTEXT INJECTED INTO DESIGN PROMPT\n"
            "=========================================================================\n"
            f"{context_section}"
            "=========================================================================\n\n"
            f"DESIGN OBJECTIVE: {prompt}"
        )
        return augmented

    def _synthesize_offline_response(
        self,
        query: str,
        retrieved: List[Union[KnowledgeItem, TrendItem]],
        trends: List[Dict[str, Any]],
        recs: List[Union[KnowledgeItem, TrendItem]]
    ) -> str:
        """Programmatically generate a rich context-aware design response."""
        response_parts = []
        response_parts.append("### Fashion AI Assistant - Generation Report")
        response_parts.append(f"Processed Query: *\"{query}\"*\n")

        # 1. Synthesize retrieved knowledge
        if retrieved:
            response_parts.append("#### Retrieved Domain Insights")
            for item in retrieved:
                citation = f"`[{item.id}]`"
                if isinstance(item, KnowledgeItem):
                    response_parts.append(
                        f"- According to category **{item.name}** ({item.category}): "
                        f"{item.content} {citation}"
                    )
                elif isinstance(item, TrendItem):
                    response_parts.append(
                        f"- Detected trend record **{item.name}** ({item.category}): "
                        f"{item.description} {citation}"
                    )
            response_parts.append("")

        # 2. Synthesize active trends
        active_matching = [
            t for t in trends 
            if any(term in t["element"] for term in query.lower().split())
        ]
        if active_matching:
            response_parts.append("#### Aligned Active Trend Mentions")
            for t in active_matching[:3]:
                response_parts.append(
                    f"- **{t['element'].capitalize()}** is highly active with a growth velocity of "
                    f"**+{t['growth_rate'] * 100:.1f}%** and {t['mentions_count']} recent logs."
                )
            response_parts.append("")

        # 3. Synthesize recommendations
        if recs:
            response_parts.append("#### Recommended Design Inspirations")
            for item, score in recs[:3]:
                citation = f"`[{item.id}]`"
                response_parts.append(
                    f"- **{item.name}** (Relevance Match: {score:.2f}) - "
                    f"{getattr(item, 'content', getattr(item, 'description', ''))[:120]}... {citation}"
                )
            response_parts.append("")

        # 4. Citations & Summary
        all_ids = [item.id for item in retrieved] + [item.id for item, _ in recs[:3]]
        unique_ids = sorted(list(set(all_ids)))
        if unique_ids:
            citations_str = ", ".join(f"`[{uid}]`" for uid in unique_ids)
            response_parts.append(f"**Citations**: This response is grounded in: {citations_str}")

        return "\n".join(response_parts)

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute the complete RAG cycle.

        Parameters
        ----------
        query_text : str
            The input design request or user search.
        top_k : int
            Number of matching documents and recommendations.

        Returns
        -------
        dict containing search outputs, prompt contexts, and response strings.
        """
        logger.info(f"RAG query received: '{query_text}' | top_k={top_k}")
        start_time = time.time()

        # 1. Retrieve relevant items
        retrieved_pairs = self.retriever.retrieve(query_text, top_k=top_k)
        retrieved_items = [item for item, score in retrieved_pairs]

        # 2. Gather active trends from tracker
        active_trends = self.trend_analyzer.get_active_trends()

        # 3. Get recommendations matching the query context
        pref_dict = {"query": query_text}
        recommendations = self.recommendation_engine.recommend(pref_dict, top_n=top_k)

        # 4. Construct augmented prompt context
        augmented_prompt = self.augment_prompt(query_text, retrieved_items)

        # 5. Synthesize grounded generation response
        generated_response = self._synthesize_offline_response(
            query=query_text,
            retrieved=retrieved_items,
            trends=active_trends,
            recs=recommendations
        )

        elapsed = time.time() - start_time
        logger.info(f"RAG query completed in {elapsed:.3f} seconds.")

        return {
            "query": query_text,
            "retrieved_items": [
                {"id": item.id, "name": item.name, "category": item.category, "score": score}
                for item, score in retrieved_pairs
            ],
            "active_trends": active_trends[:10],
            "recommendations": [
                {"id": item.id, "name": item.name, "score": score}
                for item, score in recommendations
            ],
            "augmented_prompt": augmented_prompt,
            "response": generated_response,
            "latency_seconds": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }

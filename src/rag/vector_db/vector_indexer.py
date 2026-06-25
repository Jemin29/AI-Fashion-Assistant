"""
week5/vector_db/vector_indexer.py
=================================
FAISS Vector Indexer for dense similarity searches in Week 5 RAG system.
Supports FlatL2 and InnerProduct metrics and disk persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import faiss
import numpy as np
from loguru import logger

from src.utils.config_manager import VectorDbConfig, get_default_config


class VectorIndexer:
    """
    Manages building, querying, and serializing a FAISS dense vector index.
    Supports index-to-ID mappings to resolve vector positions back to item IDs.
    """

    def __init__(
        self,
        config: Optional[VectorDbConfig] = None,
        dimension: int = 384
    ) -> None:
        """
        Initialize the Vector Indexer.

        Parameters
        ----------
        config : VectorDbConfig, optional
            Config parameters. If omitted, uses default config.
        dimension : int
            Dimensionality of target dense vectors (default 384).
        """
        if config is None:
            self.config = get_default_config().vector_db
        else:
            self.config = config

        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.item_ids: List[str] = []

        self.clear()

    def clear(self) -> None:
        """Reset index and clear ID mappings."""
        metric_type = self.config.index_type.upper()
        if metric_type == "FLATL2":
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.debug(f"Initialized FAISS IndexFlatL2 (Dimension: {self.dimension})")
        elif metric_type == "INNERPRODUCT":
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.debug(f"Initialized FAISS IndexFlatIP (Dimension: {self.dimension})")
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.config.index_type}")

        self.item_ids = []

    def add_items(self, item_ids: List[str], embeddings: np.ndarray) -> None:
        """
        Add a batch of items and their corresponding embeddings to the FAISS index.

        Parameters
        ----------
        item_ids : list of str
            Unique identifiers matching the embeddings order.
        embeddings : np.ndarray
            2D array of shape [num_items, dimension].
        """
        if len(item_ids) == 0:
            return

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}"
            )

        if len(item_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Length mismatch. IDs count: {len(item_ids)}, Embeddings count: {embeddings.shape[0]}"
            )

        # Normalize for InnerProduct (cosine similarity behavior)
        vecs_to_add = embeddings.astype(np.float32)
        if self.config.index_type.upper() == "INNERPRODUCT":
            norms = np.linalg.norm(vecs_to_add, axis=1, keepdims=True)
            vecs_to_add = vecs_to_add / (norms + 1e-9)

        # Add to FAISS Index
        self.index.add(vecs_to_add)
        self.item_ids.extend(item_ids)
        logger.info(f"Added {len(item_ids)} vectors to FAISS index. Total index count: {self.index.ntotal}")

        # Trigger auto save if enabled
        if self.config.auto_save:
            self.save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform a dense similarity search over indexed vectors.

        Parameters
        ----------
        query_embedding : np.ndarray
            1D or 2D array representing the query representation.
        top_k : int
            Max items count to return (default 5).

        Returns
        -------
        list of tuple of (item_id, score/distance)
        """
        # Prepare input shape
        query = query_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch. Expected {self.dimension}, got {query.shape[1]}"
            )

        if self.index is None or self.index.ntotal == 0:
            return []

        # Normalize query vector for InnerProduct index
        if self.config.index_type.upper() == "INNERPRODUCT":
            norms = np.linalg.norm(query, axis=1, keepdims=True)
            query = query / (norms + 1e-9)

        # Clamp top_k to total records
        k = min(top_k, self.index.ntotal)
        if k <= 0:
            return []

        # Execute search
        distances, indices = self.index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.item_ids):
                continue
            item_id = self.item_ids[idx]
            results.append((item_id, float(dist)))

        return results

    # ── Disk Serialization ───────────────────────────────────────────────────

    def save(self, storage_path: Optional[Union[str, Path]] = None) -> None:
        """
        Serialize index binaries and mapping details to disk.

        Parameters
        ----------
        storage_path : Path or str, optional
            Path on disk to load/save index partitions. Defaults to config settings.
        """
        target_dir = Path(storage_path or self.config.storage_path).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "index.faiss"
        mapping_file = target_dir / "mapping.json"

        try:
            # Save FAISS Binary
            faiss.write_index(self.index, str(index_file))
            
            # Save ID Mapping list
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(self.item_ids, f, indent=2)
                
            logger.success(f"Successfully saved FAISS index and mappings to: {target_dir}")
        except Exception as err:
            logger.error(f"Failed to save FAISS index partitions: {err}")
            raise

    def load(self, storage_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load index binaries and mapping details from disk.

        Parameters
        ----------
        storage_path : Path or str, optional
        """
        target_dir = Path(storage_path or self.config.storage_path).resolve()
        index_file = target_dir / "index.faiss"
        mapping_file = target_dir / "mapping.json"

        if not index_file.exists() or not mapping_file.exists():
            logger.warning(f"Failed to load FAISS index partitions. Files missing in: {target_dir}")
            return

        try:
            # Read index
            self.index = faiss.read_index(str(index_file))
            
            # Read mapping IDs
            with open(mapping_file, "r", encoding="utf-8") as f:
                self.item_ids = json.load(f)

            # Validate consistency
            if self.index.d != self.dimension:
                raise ValueError(
                    f"Loaded index dimension mismatch. Expected {self.dimension}, got {self.index.d}"
                )
            if self.index.ntotal != len(self.item_ids):
                raise ValueError(
                    f"Corrupted mapping index. FAISS entries: {self.index.ntotal}, IDs list: {len(self.item_ids)}"
                )

            logger.success(
                f"Successfully loaded FAISS index ({self.index.ntotal} records) from: {target_dir}"
            )
        except Exception as err:
            logger.error(f"Failed to parse saved FAISS index: {err}")
            self.clear()
            raise

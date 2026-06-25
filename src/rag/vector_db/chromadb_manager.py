"""
week5/vector_db/chromadb_manager.py
==================================
ChromaDB Manager for Week 5 vector database integration.
Exposes standard collections CRUD and query interfaces.
Includes a robust in-memory mock fallback mode if the chromadb package is missing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

# Dynamic import to support graceful fallback
try:
    import chromadb
    from chromadb.api.models.Collection import Collection as ChromaCollection
    _CHROMADB_AVAILABLE = True
except ImportError:
    _CHROMADB_AVAILABLE = False
    logger.warning("chromadb package not installed. Manager will run in mock database mode.")


# =============================================================================
# ── Mock ChromaDB Implementation (In-Memory Fallback)
# =============================================================================

class MockChromaCollection:
    """Mock implementation of a ChromaDB Collection for in-memory operations."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[Union[np.ndarray, List[List[float]]]] = None
    ) -> None:
        """Add documents to mock collection."""
        if metadatas is None:
            metadatas = [{} for _ in range(len(ids))]
        
        # Convert embeddings list to numpy array if list passed
        processed_embeddings = []
        if embeddings is not None:
            if isinstance(embeddings, list):
                processed_embeddings = [np.array(e, dtype=np.float32) for e in embeddings]
            else:
                processed_embeddings = [embeddings[i] for i in range(embeddings.shape[0])]
        else:
            # Generate mock embeddings based on document text hashes
            for doc in documents:
                sha = hashlib.sha256(doc.encode("utf-8")).hexdigest()
                seed = int(sha, 16) % (2**32)
                rng = np.random.default_rng(seed)
                vec = rng.normal(0.0, 1.0, 384)
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
                processed_embeddings.append(vec.astype(np.float32))

        for idx, item_id in enumerate(ids):
            if item_id in self.ids:
                # Update existing
                pos = self.ids.index(item_id)
                self.documents[pos] = documents[idx]
                self.metadatas[pos] = metadatas[idx]
                self.embeddings[pos] = processed_embeddings[idx]
            else:
                self.ids.append(item_id)
                self.documents.append(documents[idx])
                self.metadatas.append(metadatas[idx])
                self.embeddings.append(processed_embeddings[idx])

    def query(
        self,
        query_embeddings: Optional[Union[np.ndarray, List[List[float]]]] = None,
        query_texts: Optional[List[str]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Any]]:
        """Mock querying using cosine similarity and metadata filters."""
        if not self.ids:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Parse query embedding
        target_vec: Optional[np.ndarray] = None
        if query_embeddings is not None:
            if isinstance(query_embeddings, list):
                target_vec = np.array(query_embeddings[0], dtype=np.float32)
            else:
                # Handle 1D or 2D numpy array
                if query_embeddings.ndim == 2:
                    target_vec = query_embeddings[0].astype(np.float32)
                else:
                    target_vec = query_embeddings.astype(np.float32)
        elif query_texts is not None:
            # Hash seed mock embedding for query text
            sha = hashlib.sha256(query_texts[0].encode("utf-8")).hexdigest()
            seed = int(sha, 16) % (2**32)
            rng = np.random.default_rng(seed)
            target_vec = rng.normal(0.0, 1.0, 384)
            norm = np.linalg.norm(target_vec)
            if norm > 1e-9:
                target_vec = target_vec / norm

        # Normalize query vector
        if target_vec is not None:
            norm = np.linalg.norm(target_vec)
            if norm > 1e-9:
                target_vec = target_vec / norm
        else:
            # Fallback uniform query
            target_vec = np.ones(384, dtype=np.float32) / np.sqrt(384)

        # 1. Filter candidates by `where` metadata dict
        candidates = []
        for i in range(len(self.ids)):
            meta = self.metadatas[i]
            matched = True
            if where:
                for k, v in where.items():
                    if meta.get(k) != v:
                        matched = False
                        break
            if matched:
                candidates.append(i)

        # 2. Compute similarity (InnerProduct/Cosine)
        scored_candidates = []
        for idx in candidates:
            emb = self.embeddings[idx]
            # Normalise candidate vector
            norm = np.linalg.norm(emb)
            norm_emb = emb / (norm + 1e-9) if norm > 1e-9 else emb
            
            sim = float(np.dot(target_vec, norm_emb))
            # Map similarity to distance (higher sim -> lower distance)
            distance = 1.0 - sim
            scored_candidates.append((idx, distance))

        # Sort ascending by distance (closest first)
        scored_candidates.sort(key=lambda x: x[1])
        top_results = scored_candidates[:n_results]

        res_ids = []
        res_docs = []
        res_metas = []
        res_dists = []

        for idx, dist in top_results:
            res_ids.append(self.ids[idx])
            res_docs.append(self.documents[idx])
            res_metas.append(self.metadatas[idx])
            res_dists.append(dist)

        return {
            "ids": [res_ids],
            "documents": [res_docs],
            "metadatas": [res_metas],
            "distances": [res_dists]
        }

    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[Union[np.ndarray, List[List[float]]]] = None
    ) -> None:
        """Update existing documents."""
        for idx, item_id in enumerate(ids):
            if item_id not in self.ids:
                logger.warning(f"Mock update target ID '{item_id}' not found.")
                continue
            
            pos = self.ids.index(item_id)
            if documents is not None:
                self.documents[pos] = documents[idx]
            if metadatas is not None:
                self.metadatas[pos] = {**self.metadatas[pos], **metadatas[idx]}
            
            if embeddings is not None:
                if isinstance(embeddings, list):
                    self.embeddings[pos] = np.array(embeddings[idx], dtype=np.float32)
                else:
                    self.embeddings[pos] = embeddings[idx].astype(np.float32)

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        for item_id in ids:
            if item_id in self.ids:
                pos = self.ids.index(item_id)
                self.ids.pop(pos)
                self.documents.pop(pos)
                self.metadatas.pop(pos)
                self.embeddings.pop(pos)


class MockChromaClient:
    """Mock implementation of standard ChromaDB client."""

    def __init__(self) -> None:
        self.collections: Dict[str, MockCollection] = {}

    def get_or_create_collection(self, name: str) -> MockChromaCollection:
        if name not in self.collections:
            self.collections[name] = MockChromaCollection(name)
        return self.collections[name]

    def delete_collection(self, name: str) -> None:
        if name in self.collections:
            del self.collections[name]


# =============================================================================
# ── ChromaDB Manager
# =============================================================================

class ChromaDbManager:
    """
    Handles connections, schema indexing, and CRUD methods on ChromaDB.
    Maintains collections for fashion styles, trends, brands, recommendations, and preferences.
    """

    def __init__(
        self,
        persist_directory: Optional[Union[str, Path]] = None,
        force_mock: bool = False
    ) -> None:
        """
        Initialize the ChromaDB Manager.

        Parameters
        ----------
        persist_directory : str or Path, optional
            Path where vector indices are saved on disk. If omitted, uses in-memory.
        force_mock : bool
            Force in-memory mock client fallback.
        """
        self.force_mock = force_mock
        self.client: Union[chromadb.Client, MockChromaClient] = None
        self.is_mock_mode = False

        self._initialize_client(persist_directory)

    def _initialize_client(self, persist_directory: Optional[Union[str, Path]]) -> None:
        """Connect to database client."""
        if self.force_mock or not _CHROMADB_AVAILABLE:
            self.client = MockChromaClient()
            self.is_mock_mode = True
            logger.info("ChromaDbManager configured in-memory Mock Client fallback.")
            return

        try:
            if persist_directory:
                dir_path = Path(persist_directory).resolve()
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Connecting to persistent ChromaDB Client: {dir_path}")
                self.client = chromadb.PersistentClient(path=str(dir_path))
            else:
                logger.info("Connecting to ephemeral in-memory ChromaDB Client...")
                self.client = chromadb.EphemeralClient()
            self.is_mock_mode = False
        except Exception as err:
            logger.warning(
                f"Failed to connect to native ChromaDB Client: {err}. "
                "Switching to in-memory Mock Client fallback."
            )
            self.client = MockChromaClient()
            self.is_mock_mode = True

    # ── Collections CRUD APIs ────────────────────────────────────────────────

    def create_collection(self, collection_name: str) -> Union[ChromaCollection, MockChromaCollection]:
        """
        Register a new document collection.

        Parameters
        ----------
        collection_name : str

        Returns
        -------
        Collection object
        """
        clean_name = collection_name.lower().strip()
        logger.debug(f"Creating collection '{clean_name}'...")
        return self.client.get_or_create_collection(clean_name)

    def insert_documents(
        self,
        collection_name: str,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        """
        Add documents, metadatas, and optional embeddings to target collection.

        Parameters
        ----------
        collection_name : str
        ids : list of str
        documents : list of str
        metadatas : list of dict, optional
        embeddings : np.ndarray, optional
        """
        collection = self.create_collection(collection_name)
        
        # Convert numpy embeddings array to list format for native ChromaDB compatibility
        embs_list = None
        if embeddings is not None:
            embs_list = embeddings.tolist()

        logger.info(f"Inserting {len(ids)} documents into ChromaDB collection '{collection_name}'...")
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embs_list
        )

    def search_documents(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_embeddings: Optional[np.ndarray] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query target collection using text or vector embeddings.

        Parameters
        ----------
        collection_name : str
        query_text : str, optional
        query_embeddings : np.ndarray, optional
        n_results : int
        where : dict, optional
            Metadata filter queries.

        Returns
        -------
        list of dict representing matched documents.
        """
        collection = self.create_collection(collection_name)
        
        q_embs = None
        if query_embeddings is not None:
            if query_embeddings.ndim == 1:
                q_embs = [query_embeddings.tolist()]
            else:
                q_embs = query_embeddings.tolist()

        q_texts = [query_text] if query_text else None

        logger.debug(f"Searching ChromaDB collection '{collection_name}' | n_results={n_results} | filters={where}")
        
        raw_res = collection.query(
            query_embeddings=q_embs,
            query_texts=q_texts,
            n_results=n_results,
            where=where
        )

        # Parse ChromaDB output dict into standard flat list format
        parsed_results = []
        if raw_res and "ids" in raw_res and raw_res["ids"]:
            ids = raw_res["ids"][0]
            docs = raw_res.get("documents", [[]])[0]
            metas = raw_res.get("metadatas", [[]])[0]
            dists = raw_res.get("distances", [[]])[0]

            for idx, item_id in enumerate(ids):
                parsed_results.append({
                    "id": item_id,
                    "document": docs[idx] if idx < len(docs) else "",
                    "metadata": metas[idx] if idx < len(metas) else {},
                    "distance": float(dists[idx]) if idx < len(dists) else 0.0
                })

        return parsed_results

    def delete_documents(self, collection_name: str, ids: List[str]) -> None:
        """Delete documents by ID list."""
        collection = self.create_collection(collection_name)
        logger.info(f"Deleting {len(ids)} documents from ChromaDB collection '{collection_name}'...")
        collection.delete(ids=ids)

    def update_documents(
        self,
        collection_name: str,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        """Update existing documents, metadatas, or embeddings."""
        collection = self.create_collection(collection_name)
        
        embs_list = None
        if embeddings is not None:
            embs_list = embeddings.tolist()

        logger.info(f"Updating {len(ids)} documents in ChromaDB collection '{collection_name}'...")
        collection.update(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embs_list
        )

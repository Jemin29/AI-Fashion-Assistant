"""
week5/rag/document_ingestion.py
==============================
Fashion Document Ingestion Pipeline.
Loads documents (JSON, TXT, CSV, Markdown), chunks them, generates dense embeddings,
and stores them in ChromaDB.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.rag.vector_db.chromadb_manager import ChromaDbManager


class FashionDocumentIngester:
    """
    Ingests fashion documents of various formats (JSON, TXT, CSV, Markdown),
    chunks the text content, generates dense vectors, and loads them into ChromaDB.
    """

    def __init__(
        self,
        embedder: EmbeddingsGenerator,
        db_manager: ChromaDbManager
    ) -> None:
        """
        Initialize the Fashion Document Ingester.

        Parameters
        ----------
        embedder : EmbeddingsGenerator
            The active embedding generation engine.
        db_manager : ChromaDbManager
            ChromaDB interface manager.
        """
        self.embedder = embedder
        self.db_manager = db_manager
        logger.info("FashionDocumentIngester successfully initialized.")

    def load_document(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load text and metadata from a file based on its extension.

        Parameters
        ----------
        file_path : str or Path

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionary records containing "text" and "metadata" keys.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found at: {path}")

        suffix = path.suffix.lower()
        logger.info(f"Loading document '{path.name}' with format '{suffix}'...")

        if suffix == ".json":
            return self._load_json(path)
        elif suffix == ".txt":
            return self._load_txt(path)
        elif suffix == ".csv":
            return self._load_csv(path)
        elif suffix in (".md", ".markdown"):
            return self._load_markdown(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse JSON file as lists of items or a structured record."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        records = []
        if isinstance(content, list):
            for idx, item in enumerate(content):
                if isinstance(item, dict):
                    # Extract text content from common text-like keys
                    text_content = ""
                    for key in ("text", "content", "description", "body"):
                        if key in item and item[key]:
                            text_content = str(item[key])
                            break
                    if not text_content:
                        # Fallback: join key-values
                        text_content = ", ".join(f"{k}: {v}" for k, v in item.items() if v is not None)
                    
                    metadata = {**item, "source": file_path.name, "item_index": idx}
                    records.append({"text": text_content, "metadata": metadata})
                else:
                    records.append({
                        "text": str(item),
                        "metadata": {"source": file_path.name, "item_index": idx}
                    })
        elif isinstance(content, dict):
            text_content = ""
            for key in ("text", "content", "description", "body"):
                if key in content and content[key]:
                    text_content = str(content[key])
                    break
            if not text_content:
                text_content = ", ".join(f"{k}: {v}" for k, v in content.items() if v is not None)
            
            metadata = {**content, "source": file_path.name}
            records.append({"text": text_content, "metadata": metadata})
        else:
            records.append({
                "text": str(content),
                "metadata": {"source": file_path.name}
            })

        return records

    def _load_txt(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse raw text file, splitting it by double newlines into paragraphs."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        records = []
        for idx, para in enumerate(paragraphs):
            records.append({
                "text": para,
                "metadata": {
                    "source": file_path.name,
                    "paragraph_index": idx
                }
            })
        return records

    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse CSV rows into formatted text and metadata records."""
        records = []
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                text_content = ""
                for key in ("text", "content", "description", "body"):
                    if key in row and row[key]:
                        text_content = str(row[key])
                        break
                if not text_content:
                    text_content = ", ".join(f"{k}: {v}" for k, v in row.items() if v)

                metadata = {**row, "source": file_path.name, "row_index": idx}
                records.append({"text": text_content, "metadata": metadata})
        return records

    def _load_markdown(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse Markdown file, separating content sections based on headers."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = []
        current_header = "Intro"
        current_level = 0
        current_lines: List[str] = []

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                parts = stripped.split(" ", 1)
                hashes = parts[0]
                if all(c == "#" for c in hashes) and len(parts) > 1:
                    # Save the accumulated section content
                    text_content = "\n".join(current_lines).strip()
                    if text_content:
                        sections.append({
                            "text": text_content,
                            "metadata": {
                                "source": file_path.name,
                                "section_header": current_header,
                                "header_level": current_level
                            }
                        })
                    current_header = parts[1].strip()
                    current_level = len(hashes)
                    current_lines = []
                    continue
            current_lines.append(line)

        # Save last section
        text_content = "\n".join(current_lines).strip()
        if text_content:
            sections.append({
                "text": text_content,
                "metadata": {
                    "source": file_path.name,
                    "section_header": current_header,
                    "header_level": current_level
                }
            })

        return sections

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split a block of text into sliding-window chunks with configurable size and overlap.

        Parameters
        ----------
        text : str
            The input string text content.
        chunk_size : int
            Character length boundary of each chunk.
        chunk_overlap : int, optional
            Overlap length between subsequent chunks. If omitted, defaults dynamically to chunk_size // 10.
        metadata : dict, optional
            Base metadata dictionary to attach to each chunk.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionary chunks with "text" and "metadata" keys.
        """
        if metadata is None:
            metadata = {}

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive and greater than zero.")

        if chunk_overlap is None:
            chunk_overlap = min(50, max(0, chunk_size // 10))

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative.")

        # If text fits within chunk size, return it in one piece
        if len(text) <= chunk_size:
            return [{
                "text": text,
                "metadata": {
                    **metadata,
                    "chunk_index": 0,
                    "start_char": 0,
                    "end_char": len(text)
                }
            }]

        chunks = []
        start = 0
        step = chunk_size - chunk_overlap

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append({
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": len(chunks),
                    "start_char": start,
                    "end_char": end
                }
            })
            if end == len(text):
                break
            start += step

        return chunks

    def ingest_file(
        self,
        file_path: Union[str, Path],
        collection_name: str,
        chunk_size: int = 500,
        chunk_overlap: Optional[int] = None
    ) -> List[str]:
        """
        Load, chunk, embed, and store a document in ChromaDB.

        Parameters
        ----------
        file_path : str or Path
        collection_name : str
            ChromaDB target collection name.
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        List[str]
            List of generated unique document/chunk IDs inserted.
        """
        # 1. Load document records
        records = self.load_document(file_path)

        # 2. Chunk text contents
        all_chunks = []
        for rec in records:
            chunks = self.chunk_text(
                text=rec["text"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=rec["metadata"]
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning(f"No text contents extracted or chunked from file: {file_path}")
            return []

        # 3. Generate unique IDs and extract values for ingestion
        path_hash = hashlib.sha256(str(Path(file_path).resolve()).encode("utf-8")).hexdigest()[:16]
        ids = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(all_chunks):
            # Formulate stable, unique ID using path hash and chunk index
            chunk_id = f"doc_{path_hash}_chunk_{idx}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])

        # 4. Generate embeddings
        logger.info(f"Computing embeddings for {len(documents)} text chunks from '{Path(file_path).name}'...")
        embeddings = self.embedder.embed_batch(documents)

        # 5. Insert into ChromaDB
        logger.info(f"Upserting chunks into ChromaDB collection '{collection_name}'...")
        self.db_manager.insert_documents(
            collection_name=collection_name,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return ids

    def ingest_directory(
        self,
        dir_path: Union[str, Path],
        collection_name: str,
        chunk_size: int = 500,
        chunk_overlap: Optional[int] = None,
        file_extensions: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Scan a directory recursively and ingest all files matching specific extensions.

        Parameters
        ----------
        dir_path : str or Path
        collection_name : str
        chunk_size : int
        chunk_overlap : int
        file_extensions : List[str], optional
            Filters to target specific formats. Defaults to JSON, TXT, CSV, and Markdown.

        Returns
        -------
        Dict[str, List[str]]
            Mapping of file paths to lists of chunk IDs ingested.
        """
        if file_extensions is None:
            file_extensions = [".json", ".txt", ".csv", ".md", ".markdown"]

        dir_p = Path(dir_path).resolve()
        if not dir_p.exists() or not dir_p.is_dir():
            raise ValueError(f"Directory path does not exist or is not a directory: {dir_path}")

        logger.info(f"Scanning directory '{dir_p}' for extensions: {file_extensions}...")

        # Normalize extensions to start with dot
        normalized_exts = []
        for ext in file_extensions:
            ext_clean = ext.strip().lower()
            if not ext_clean.startswith("."):
                ext_clean = f".{ext_clean}"
            normalized_exts.append(ext_clean)

        ingested_files = {}

        for ext in normalized_exts:
            for file_p in dir_p.glob(f"**/*{ext}"):
                if file_p.is_file():
                    file_key = str(file_p.resolve().as_posix())
                    try:
                        logger.info(f"Processing auto-discovered file: {file_p.name}")
                        chunk_ids = self.ingest_file(
                            file_path=file_p,
                            collection_name=collection_name,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        ingested_files[file_key] = chunk_ids
                    except Exception as err:
                        logger.error(f"Failed to ingest file '{file_p}': {err}")

        return ingested_files

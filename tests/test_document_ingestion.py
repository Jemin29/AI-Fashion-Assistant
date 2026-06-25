"""
week5/tests/test_document_ingestion.py
======================================
Unit tests for the Fashion Document Ingestion Pipeline.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.rag.document_ingestion import FashionDocumentIngester
from src.rag.vector_db.chromadb_manager import ChromaDbManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files, cleaning up automatically."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_embedder():
    """Returns an EmbeddingsGenerator initialized in mock mode."""
    return EmbeddingsGenerator(force_mock=True)


@pytest.fixture
def mock_db_manager():
    """Returns a ChromaDbManager initialized in mock mode."""
    return ChromaDbManager(force_mock=True)


class TestFashionDocumentIngester:
    """Validate parsing, chunking, embedding generation, and ChromaDB insertion pipelines."""

    def test_ingester_initialization(self, mock_embedder, mock_db_manager):
        """Verify the ingester initializes correctly."""
        ingester = FashionDocumentIngester(embedder=mock_embedder, db_manager=mock_db_manager)
        assert ingester.embedder == mock_embedder
        assert ingester.db_manager == mock_db_manager

    def test_load_document_json(self, temp_dir, mock_embedder, mock_db_manager):
        """Verify loading and parsing JSON files."""
        ingester = FashionDocumentIngester(embedder=mock_embedder, db_manager=mock_db_manager)

        # 1. JSON List of Dictionaries
        list_dict_path = temp_dir / "list_dict.json"
        list_dict_data = [
            {"text": "Oversized cargo pants are trending.", "category": "streetwear"},
            {"content": "Silk slip dresses in pastel hues.", "category": "luxury"},
            {"custom_field": "Velvet boots.", "color": "black"}
        ]
        with open(list_dict_path, "w", encoding="utf-8") as f:
            json.dump(list_dict_data, f)

        records = ingester.load_document(list_dict_path)
        assert len(records) == 3
        assert records[0]["text"] == "Oversized cargo pants are trending."
        assert records[0]["metadata"]["category"] == "streetwear"
        assert records[0]["metadata"]["source"] == "list_dict.json"
        assert records[1]["text"] == "Silk slip dresses in pastel hues."
        assert records[2]["text"] == "custom_field: Velvet boots., color: black"

        # 2. JSON Single Dictionary
        single_dict_path = temp_dir / "single_dict.json"
        single_dict_data = {"description": "Linen shirts for summer.", "fabric": "linen"}
        with open(single_dict_path, "w", encoding="utf-8") as f:
            json.dump(single_dict_data, f)

        records_single = ingester.load_document(single_dict_path)
        assert len(records_single) == 1
        assert records_single[0]["text"] == "Linen shirts for summer."
        assert records_single[0]["metadata"]["fabric"] == "linen"

        # 3. JSON List of Strings/Other
        list_str_path = temp_dir / "list_str.json"
        list_str_data = ["Streetwear hoodies.", 12345]
        with open(list_str_path, "w", encoding="utf-8") as f:
            json.dump(list_str_data, f)

        records_str = ingester.load_document(list_str_path)
        assert len(records_str) == 2
        assert records_str[0]["text"] == "Streetwear hoodies."
        assert records_str[1]["text"] == "12345"

        # 4. JSON Raw String/Value fallback
        raw_val_path = temp_dir / "raw_val.json"
        with open(raw_val_path, "w", encoding="utf-8") as f:
            json.dump("Just a raw string in JSON.", f)

        records_raw = ingester.load_document(raw_val_path)
        assert len(records_raw) == 1
        assert records_raw[0]["text"] == "Just a raw string in JSON."

    def test_load_document_txt(self, temp_dir, mock_embedder, mock_db_manager):
        """Verify loading and parsing TXT files into paragraphs."""
        ingester = FashionDocumentIngester(embedder=mock_embedder, db_manager=mock_db_manager)

        txt_path = temp_dir / "sample.txt"
        txt_content = (
            "Paragraph one is about winter trends.\n"
            "Heavy woolen coats and layering techniques.\n\n"
            "Paragraph two describes summer footwear.\n"
            "Strappy sandals and canvas sneakers.\n\n"
            "  \n\n"  # Empty paragraph block
            "Paragraph three concludes with spring accessories."
        )
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

        records = ingester.load_document(txt_path)
        assert len(records) == 3
        assert records[0]["text"] == "Paragraph one is about winter trends.\nHeavy woolen coats and layering techniques."
        assert records[0]["metadata"]["paragraph_index"] == 0
        assert records[1]["text"] == "Paragraph two describes summer footwear.\nStrappy sandals and canvas sneakers."
        assert records[1]["metadata"]["paragraph_index"] == 1
        assert records[2]["text"] == "Paragraph three concludes with spring accessories."
        assert records[2]["metadata"]["paragraph_index"] == 2

    def test_load_document_csv(self, temp_dir, mock_embedder, mock_db_manager):
        """Verify loading and parsing CSV files."""
        ingester = FashionDocumentIngester(embedder=mock_embedder, db_manager=mock_db_manager)

        csv_path = temp_dir / "sample.csv"
        csv_data = [
            {"text": "Chunky sneakers dominate streetwear.", "popularity": "high"},
            {"description": "Monochrome tailoring is elegant.", "popularity": "medium"},
            {"style": "Athleisure", "popularity": "low"}
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "description", "style", "popularity"])
            writer.writeheader()
            for row in csv_data:
                # Fill missing columns to avoid KeyError
                row_full = {
                    "text": row.get("text", ""),
                    "description": row.get("description", ""),
                    "style": row.get("style", ""),
                    "popularity": row.get("popularity", "")
                }
                writer.writerow(row_full)

        records = ingester.load_document(csv_path)
        assert len(records) == 3
        assert records[0]["text"] == "Chunky sneakers dominate streetwear."
        assert records[0]["metadata"]["popularity"] == "high"
        assert records[0]["metadata"]["row_index"] == 0

        assert records[1]["text"] == "Monochrome tailoring is elegant."
        assert records[1]["metadata"]["popularity"] == "medium"

        # Column concatenation fallback test
        assert "style: Athleisure" in records[2]["text"]
        assert "popularity: low" in records[2]["text"]

    def test_load_document_markdown(self, temp_dir, mock_embedder, mock_db_manager):
        """Verify loading and parsing Markdown files."""
        ingester = FashionDocumentIngester(embedder=mock_embedder, db_manager=mock_db_manager)

        md_path = temp_dir / "sample.md"
        md_content = (
            "Intro paragraph without a header.\n"
            "Second line of intro.\n"
            "# Section 1: Streetwear\n"
            "Streetwear content block.\n"
            "## Sub Section: Sneakers\n"
            "Sneakers details.\n"
            "# Section 2: Luxury\n"
            "Luxury gowns and couture details."
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        records = ingester.load_document(md_path)
        assert len(records) == 4

        # Intro block
        assert records[0]["metadata"]["section_header"] == "Intro"
        assert records[0]["metadata"]["header_level"] == 0
        assert "Intro paragraph without a header." in records[0]["text"]

        # H1 Section 1
        assert records[1]["metadata"]["section_header"] == "Section 1: Streetwear"
        assert records[1]["metadata"]["header_level"] == 1
        assert records[1]["text"] == "Streetwear content block."

        # H2 Sub Section
        assert records[2]["metadata"]["section_header"] == "Sub Section: Sneakers"
        assert records[2]["metadata"]["header_level"] == 2
        assert records[2]["text"] == "Sneakers details."

        # H1 Section 2
        assert records[3]["metadata"]["section_header"] == "Section 2: Luxury"
        assert records[3]["metadata"]["header_level"] == 1
        assert records[3]["text"] == "Luxury gowns and couture details."

    def test_chunk_text(self, mock_embedder, mock_db_manager):
        """Verify the text sliding-window chunking logic."""
        ingester = FashionDocumentIngester(embedder=mock_embedder, db_manager=mock_db_manager)

        text = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
        
        # 1. Fits in a single chunk
        chunks_single = ingester.chunk_text(text, chunk_size=50)
        assert len(chunks_single) == 1
        assert chunks_single[0]["text"] == text
        assert chunks_single[0]["metadata"]["chunk_index"] == 0
        assert chunks_single[0]["metadata"]["start_char"] == 0
        assert chunks_single[0]["metadata"]["end_char"] == 26

        # 2. Split with overlap
        chunks_overlap = ingester.chunk_text(text, chunk_size=10, chunk_overlap=2, metadata={"tag": "test"})
        # Steps:
        # Step size = 10 - 2 = 8
        # Chunk 0: [0:10] = "abcdefghij"
        # Chunk 1: [8:18] = "ijklmnopqr"
        # Chunk 2: [16:26] = "qrstuvwxyz"
        assert len(chunks_overlap) == 3
        assert chunks_overlap[0]["text"] == "abcdefghij"
        assert chunks_overlap[1]["text"] == "ijklmnopqr"
        assert chunks_overlap[2]["text"] == "qrstuvwxyz"
        assert all(c["metadata"]["tag"] == "test" for c in chunks_overlap)
        assert chunks_overlap[1]["metadata"]["start_char"] == 8
        assert chunks_overlap[1]["metadata"]["end_char"] == 18
        assert chunks_overlap[2]["metadata"]["chunk_index"] == 2

        # 3. Invalid inputs validation
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ingester.chunk_text(text, chunk_size=0)

        with pytest.raises(ValueError, match="chunk_overlap must be strictly less than chunk_size"):
            ingester.chunk_text(text, chunk_size=10, chunk_overlap=10)

        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            ingester.chunk_text(text, chunk_size=10, chunk_overlap=-1)

    def test_ingest_file(self, temp_dir, mock_embedder, mock_db_manager):
        """Verify the complete load -> chunk -> embed -> ChromaDB ingestion loop."""
        ingester = FashionDocumentIngester(embedder=mock_embedder, db_manager=mock_db_manager)

        txt_path = temp_dir / "ingest_sample.txt"
        txt_content = (
            "This is paragraph one content. " * 10 + "\n\n" +  # Long paragraph
            "This is paragraph two content. " * 10
        )
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

        # Chunk size is small to force multiple chunks
        collection_name = "test_ingestion_collection"
        chunk_ids = ingester.ingest_file(
            file_path=txt_path,
            collection_name=collection_name,
            chunk_size=100,
            chunk_overlap=10
        )

        assert len(chunk_ids) > 0
        
        # Verify inserted in mock ChromaDB
        search_res = mock_db_manager.search_documents(
            collection_name=collection_name,
            n_results=100
        )
        assert len(search_res) == len(chunk_ids)
        inserted_ids = {item["id"] for item in search_res}
        assert inserted_ids == set(chunk_ids)
        for item in search_res:
            assert "source" in item["metadata"]
            assert item["metadata"]["source"] == "ingest_sample.txt"

    def test_ingest_directory(self, temp_dir, mock_embedder, mock_db_manager):
        """Verify recursive directory scanning and ingestion of all matching files."""
        ingester = FashionDocumentIngester(embedder=mock_embedder, db_manager=mock_db_manager)

        # Create nested directory structure
        sub_dir = temp_dir / "sub_folder"
        sub_dir.mkdir()

        # Write different file formats
        # 1. JSON
        json_path = temp_dir / "doc1.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"text": "JSON content data", "category": "trends"}, f)

        # 2. TXT
        txt_path = sub_dir / "doc2.TXT"  # Uppercase suffix
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("TXT content data")

        # 3. CSV
        csv_path = sub_dir / "doc3.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "tag"])
            writer.writeheader()
            writer.writerow({"text": "CSV content data", "tag": "test"})

        # 4. Markdown
        md_path = temp_dir / "doc4.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Markdown section\nContent here.")

        # 5. Unsupported file type (should be skipped)
        bin_path = temp_dir / "doc5.bin"
        with open(bin_path, "wb") as f:
            f.write(b"Binary data")

        collection = "dir_collection"
        ingested = ingester.ingest_directory(
            dir_path=temp_dir,
            collection_name=collection,
            chunk_size=200,
            chunk_overlap=20
        )

        # Assert all 4 supported files were processed
        assert len(ingested) == 4
        
        # Verify absolute path keys in mapping
        json_key = str(json_path.resolve().as_posix())
        txt_key = str(txt_path.resolve().as_posix())
        csv_key = str(csv_path.resolve().as_posix())
        md_key = str(md_path.resolve().as_posix())
        bin_key = str(bin_path.resolve().as_posix())

        assert json_key in ingested
        assert txt_key in ingested
        assert csv_key in ingested
        assert md_key in ingested
        assert bin_key not in ingested

        # Verify documents can be retrieved from mock database
        search_res = mock_db_manager.search_documents(collection_name=collection, n_results=100)
        assert len(search_res) >= 4

    def test_unsupported_and_missing_files(self, temp_dir, mock_embedder, mock_db_manager):
        """Verify proper exceptions are raised for missing files or unsupported extensions."""
        ingester = FashionDocumentIngester(embedder=mock_embedder, db_manager=mock_db_manager)

        # 1. Missing file path
        missing_path = temp_dir / "non_existent.txt"
        with pytest.raises(FileNotFoundError):
            ingester.load_document(missing_path)

        # 2. Unsupported extension
        unsupported_path = temp_dir / "sample.pdf"
        with open(unsupported_path, "w", encoding="utf-8") as f:
            f.write("PDF mock content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            ingester.load_document(unsupported_path)

        # 3. Missing directory validation
        with pytest.raises(ValueError, match="Directory path does not exist"):
            ingester.ingest_directory(temp_dir / "non_existent_folder", "col")

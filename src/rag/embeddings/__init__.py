"""
week5/embeddings
================
Embeddings generators and transformers models wrappers.
"""
from src.rag.embeddings.embeddings_generator import EmbeddingsGenerator
from src.rag.embeddings.fashion_embeddings import FashionEmbeddingEngine

__all__ = ["EmbeddingsGenerator", "FashionEmbeddingEngine"]



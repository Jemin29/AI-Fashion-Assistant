"""
week5/rag
=========
RAG prompting context and generation coupling models.
"""
from src.rag.fashion_rag import FashionRAG
from src.rag.document_ingestion import FashionDocumentIngester
from src.rag.fashion_rag_pipeline import FashionRAGPipeline
from src.rag.fashion_assistant import FashionAssistant
from src.evaluation.rag_evaluator import RAGEvaluator

__all__ = [
    "FashionRAG",
    "FashionDocumentIngester",
    "FashionRAGPipeline",
    "FashionAssistant",
    "RAGEvaluator",
]


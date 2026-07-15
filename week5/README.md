# Week 5 — Retrieval-Augmented Generation (RAG) & Recommendations

> **AI-Powered Fashion Design Assistant · Week 5 Deliverables**  
> Context-aware domain intelligence, vector database curation (ChromaDB + FAISS), CLIP embeddings, hybrid search, style recommendations, and trend forecasting.

---

## 📁 Overview

Week 5 introduces **Retrieval-Augmented Generation (RAG) & Recommendations**, providing semantic, grounded search capabilities to the virtual assistant and matching users with styles and brands based on user interaction profiles and seasonal trend forecasting.

To keep the repository clean and avoid duplicating code, all business logic resides in [src/rag/](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/rag), [src/recommendations/](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/recommendations), and [src/trends/](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/trends).

---

## 📁 Directory Structure

```
week5/
├── assets/                  # DB ingestion schemas and trend diagrams
│   └── .gitkeep
├── results/                 # Evaluation hit-rate statistics and sample RAG outputs
│   └── .gitkeep
├── README.md                # This overview guide
├── architecture.md          # Vector DB indexing and hybrid retrieval flow diagrams
└── documentation.md         # API guides, queries, and vector stores configuration
```

---

## 🚀 Key Deliverables Reused from `src/`

All deliverables are fully implemented within the repository's canonical packages:

1. **CLIP Embeddings**: Implemented in [src/rag/embeddings/embeddings_generator.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/rag/embeddings/embeddings_generator.py) and [fashion_embeddings.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/rag/embeddings/fashion_embeddings.py). Encodes text queries and image representations into uniform 384D coordinate vectors.
2. **Vector DB (ChromaDB & FAISS)**: Implemented in [src/rag/vector_db/chromadb_manager.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/rag/vector_db/chromadb_manager.py) and [vector_indexer.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/rag/vector_db/vector_indexer.py). ChromaDB acts as the conversational and trends collection store; FAISS serves as high-speed vector retrieval.
3. **Fashion Knowledge Base & QA**: Grounded in JSON files under [outputs/knowledge_base/](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/outputs/knowledge_base/) and indexed in the vector store.
4. **Hybrid Retrieval**: Implemented in [src/rag/retrieval/hybrid_retriever.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/rag/retrieval/hybrid_retriever.py). Combines BM25 keyword matching with dense vector similarity search.
5. **Recommendation Engine**: Implemented in [src/recommendations/recommendation_engine.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/recommendations/recommendation_engine.py). Recommends styles and brands by matching generated history vectors against target styles and active brand profiles.
6. **Trend Forecasting**: Implemented in [src/trends/trend_forecaster.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/trends/trend_forecaster.py) and [trend_analyzer.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/trends/trend_analyzer.py). Traces category velocities and seasonally projects popular silhouettes/colors.

---

## ⚙️ Quick Start

Verify the RAG and recommendation classes can load in your environment:

```bash
# Verify RAG components load successfully
python -c "from src.rag.fashion_rag import FashionRAG; print('RAG Pipeline Coordinator successfully loaded!')"

# Verify Recommendations components load successfully
python -c "from src.recommendations.recommendation_engine import RecommendationEngine; print('Recommendation Engine successfully loaded!')"
```

For Python usage snippets, refer to [documentation.md](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week5/documentation.md).

# Week 5 — API & Usage Documentation

This document describes how to programmatically query the RAG pipeline, ingest custom domain documents, calculate recommendations, and analyze style trends.

---

## 🛠️ Python API Reference

### 1. Context-Aware Fashion RAG Query

Query the retriever to fetch grounded sources and generate structured responses:

```python
from src.rag.fashion_rag import FashionRAG

# Initialize the pipeline coordinator
rag = FashionRAG(
    db_path="outputs/vector_db/faiss_index",
    kb_path="outputs/knowledge_base/fashion_knowledge_base.json"
)

# Run a hybrid search query
results = rag.search(
    query="What is the difference between Egyptian cotton and linen for summer clothes?",
    top_k=3
)

for citation in results["citations"]:
    print(f"[{citation['score']:.2f}] Source: {citation['source']}")
    print(f"Content: {citation['text']}\n")
```

### 2. Live ChromaDB Collections Queries

Manage dynamic document collections inside the vectorized database:

```python
from src.rag.vector_db.chromadb_manager import ChromaDbManager

# Initialize local in-memory or persisted client
db_manager = ChromaDbManager(persist_directory="outputs/vector_db/chromadb")

# Retrieve collection entries matching criteria
style_docs = db_manager.query_collection(
    collection_name="fashion_styles",
    query_text="bohemian silhouette flowy",
    n_results=2
)
print(style_docs)
```

### 3. Generate Personal Recommendations

Generate user-focused style profiles and run the matching engine:

```python
from src.recommendations.recommendation_engine import RecommendationEngine

# Load recommendation matching layers
engine = RecommendationEngine(
    profiles_path="outputs/recommendations/user_profile.json"
)

# Retrieve recommendations for a specific user ID
rec_list = engine.get_recommendations(
    user_id="user_01",
    limit=5
)

for rec in rec_list:
    print(f"Recommended Style: {rec['style']} | Fit Score: {rec['score']:.2f}")
```

### 4. Fetch Trend Metrics

```python
from src.trends.trend_forecaster import TrendForecaster

forecaster = TrendForecaster(trends_path="outputs/knowledge_base/trend_dataset.json")

# Retrieve velocity rankings for the current season
trending = forecaster.get_trending_categories(limit=3)
for trend in trending:
    print(f"Category: {trend['name']} | Velocity: {trend['velocity']:.2f}% | Direction: {trend['status']}")
```

---

## 📥 Ingesting Custom Knowledge Bases

To load a raw `.json` file of Q&A cards or wiki pages into the vector store:

```bash
# Ingest raw text files or JSON tables into the FAISS index
python src/rag/document_ingestion.py --input_file path/to/fashion_wiki.json --output_db outputs/vector_db/faiss_index
```

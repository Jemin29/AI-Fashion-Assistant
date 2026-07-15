# Week 5 — System Architecture

This document describes the architectural flow of the **Retrieval-Augmented Generation (RAG) & Recommendation Engine** layer.

---

## 💬 RAG Information Flow

The RAG pipeline provides grounded domain knowledge to the conversational assistant. It ensures responses are supported by verified fashion literature, avoiding hallucinated recommendations.

```mermaid
graph TD
    A[Textual Documents / Datasets] --> B[Document Ingestor]
    B -->|Chunking & Preprocessing| C[Embeddings Generator]
    C -->|dense 384D float vectors| D[Vector Databases]
    D -->|FAISS Index| E[Hybrid Retriever]
    D -->|ChromaDB Collections| E
    F[User Query] --> G[Hybrid Retriever]
    G -->|Dense Similarity| H[Vector Search]
    G -->|Sparse Match| I[BM25 Keyword Search]
    H & I --> J[Merged & Re-ranked Citations]
    J --> K[LLM / Response Synthesizer]
    K --> L[Grounded Answer with Sources]
```

---

## 👗 Recommendation Matching Engine

The recommendation system consists of two parts:
1. **Style Recommender**: Maps visual preferences, color choices, and silhouettes.
2. **Brand Recommender**: Matches budget, aesthetics, and values with brand profiles (Nike, Gucci, Zara, H&M).

```mermaid
flowchart LR
    A[User Profile] --> C[Recommendation Engine]
    B[Generation History] --> C
    C --> D[Style Recommender]
    C --> E[Brand Recommender]
    D -->|Cosine Similarity| F[Style Candidates]
    E -->|Matrix Matching| G[Brand Candidates]
    F & G --> H[Ensemble Scorer]
    H --> I[Final Recommendation List]
```

- **User Profile Manager**: Manages session state and persists historical interaction vectors in a local JSON storage (`outputs/recommendations/user_profile.json`).
- **Embedding Alignment**: Computes cosine similarity between user prompt embeddings and category index profiles to rank interest scores across 12+ styles.

---

## 📈 Trend Forecasting Engine

The Trend Forecasting module uses a statistical time-series model to calculate the velocity and acceleration of fashion category popularity metrics.
- **Velocity Tracker**: Measures week-over-week growth rate of search query matches.
- **Seasonality Offset**: Adjusts scores dynamically based on the current system date (spring/summer vs autumn/winter weights).
- **Faiss Storage**: Indexes trend nodes to allow instant semantic similarity mapping against newly generated user prompt requests.

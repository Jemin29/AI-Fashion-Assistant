# Week 5 Environment & System Setup Guide

This guide describes the dependencies, configurations, and verification steps required to set up the **Fashion Intelligence and Retrieval-Augmented Generation (RAG) System** environment.

---

## ⚙️ Installation Procedures

### 1. Core Python Packages
Ensure you are operating inside the project's virtual environment:
```bash
pip install -r week5/requirements.txt
```

### 2. FAISS Vector Database Installation
Our system uses the **Facebook AI Similarity Search (FAISS)** library to index fashion article embeddings and run nearest-neighbor dense retrievals.
* **CPU Version (Default & Recommended for local setups)**:
  ```bash
  pip install faiss-cpu>=1.7.4
  ```
* **GPU-enabled Version (For CUDA production servers)**:
  ```bash
  pip install faiss-gpu>=1.7.4
  ```

### 3. Sentence Transformers
Dense text embeddings are generated using pre-trained sentence transformers (e.g. `all-MiniLM-L6-v2` by default):
```bash
pip install sentence-transformers>=2.2.2
```

---

## 🛠️ Verification Checks

To verify that the installation is complete and components are working, run the initial unit tests:
```bash
python week5/tests/run_week5_tests.py
```
This executes Pydantic validations, configuration parameters checking, and Loguru logging sink setups, ensuring that Week 5 is ready for implementation.

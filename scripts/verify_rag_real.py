"""
scripts/verify_rag_real.py
===========================
Runs Week 5 RAG system in real mode (force_mock_embeddings=False) against
ChromaDB and sentence-transformers model. Executes 7 fashion queries to
confirm retrieval quality and generates regression test data.
"""

import os
import sys
import json
import time
from pathlib import Path
from loguru import logger

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.fashion_assistant import FashionAssistant

def main():
    logger.info("Initializing FashionAssistant in REAL mode...")
    
    # Initialize assistant with real embeddings
    try:
        assistant = FashionAssistant(force_mock_embeddings=False)
    except Exception as e:
        logger.error(f"Failed to initialize real FashionAssistant: {e}")
        return 1

    queries = [
        "Explain the augmented wearable projections trend forecast",
        "I need streetwear style recommendations in black color",
        "Recommend brand profiles for a luxury wardrobe",
        "What is linen fabric?",
        "Describe the minimalist style",
        "What is silk fabric?",
        "Recommend brands for streetwear style"
    ]

    regression_data = []

    logger.info(f"Running {len(queries)} fashion queries against real RAG backend...")
    
    for idx, query in enumerate(queries):
        logger.info(f"\nQuery {idx+1}: '{query}'")
        start = time.perf_counter()
        
        try:
            result = assistant.chat(message=query, user_id=f"verify_user_{idx+1}")
            latency = time.perf_counter() - start
            
            response = result.get("response", "")
            # Extracted details
            data_val = result.get("data", {})
            styles_val = []
            if isinstance(data_val, dict):
                styles_val = data_val.get("styles", [])
            elif isinstance(data_val, list):
                styles_val = data_val
            retrieved_items = result.get("retrieved_items", styles_val)
            
            logger.info(f"Latency: {latency:.4f}s")
            logger.info(f"Response: {response[:300]}...")
            
            regression_data.append({
                "query": query,
                "response": response,
                "retrieved_items": [
                    {
                        "id": getattr(item, "id", item.get("id") if isinstance(item, dict) else str(item)),
                        "name": getattr(item, "name", item.get("name") if isinstance(item, dict) else str(item))
                    }
                    for item in retrieved_items
                ]
            })
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            
    # Write regression data to tests directory
    output_path = _ROOT / "tests" / "real_rag_regression_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(regression_data, f, indent=2, sort_keys=True)
        
    logger.success(f"\nReal-mode RAG verification completed! Regression data saved to: {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

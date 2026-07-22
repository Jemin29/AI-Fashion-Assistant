"""
tests/test_rag_regression.py
=============================
Regression tests verifying that the RAG pipeline operates in real mode,
retrieving actual responses from ChromaDB rather than falling back to mocks.
"""

import json
from pathlib import Path
import pytest
from src.rag.fashion_assistant import FashionAssistant

@pytest.fixture(scope="module")
def assistant():
    # Load FashionAssistant in real mode
    return FashionAssistant(force_mock_embeddings=False)

def test_rag_real_regression(assistant):
    # Load regression data
    regression_path = Path(__file__).resolve().parent / "real_rag_regression_data.json"
    assert regression_path.exists(), "Regression data file not found!"

    with open(regression_path, "r", encoding="utf-8") as f:
        regression_data = json.load(f)

    for item in regression_data:
        query = item["query"]
        expected_response = item["response"]

        # Run query through assistant
        result = assistant.chat(message=query, user_id="test_regression_user")
        response = result.get("response", "")

        # Verifications
        assert response, f"Empty response received for query: '{query}'"
        assert "Mock Mode" not in response, f"Mock fallback detected for query: '{query}'"
        assert "Fashion Intelligence (Mock Mode)" not in response, f"Mock response detected for query: '{query}'"
        
        # Verify response matches or contains essential structural phrases from the real db
        expected_first_line = expected_response.split("\n")[0]
        assert expected_first_line in response, f"Expected response structure '{expected_first_line}' not found in actual response: '{response}'"

from __future__ import annotations

import time
from fastapi.testclient import TestClient
from week7.backend.main import app

client = TestClient(app)


def test_start_task_success():
    """Verify posting to start a task initializes and runs eager Celery execution."""
    payload = {
        "task_type": "generation",
        "payload": {
            "prompt": "A beautiful silk dress",
            "seed": 42
        }
    }
    response = client.post("/task/start", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "task_id" in data
    assert data["task_type"] == "generation"
    # Because Celery is configured in eager mode, it executes synchronously.
    # Therefore, the task will already be completed when we poll it!
    task_id = data["task_id"]

    # Poll status
    get_resp = client.get(f"/task/{task_id}")
    assert get_resp.status_code == 200
    meta = get_resp.json()
    assert meta["task_id"] == task_id
    assert meta["status"] == "SUCCESS"
    assert meta["progress"] == 100
    assert meta["execution_time_s"] >= 0.0
    assert "result" in meta
    assert "image" in meta["result"]


def test_start_task_invalid_type():
    """Verify starting an invalid task type returns a 400."""
    payload = {
        "task_type": "invalid_type",
        "payload": {}
    }
    response = client.post("/task/start", json=payload)
    assert response.status_code == 400
    assert "Unsupported task type" in response.json()["error"]["message"]


def test_get_nonexistent_task():
    """Verify polling status of nonexistent task returns 404."""
    response = client.get("/task/non-existent-task-id-12345")
    assert response.status_code == 404


def test_delete_task_lifecycle():
    """Verify starting, polling, and deleting a task works correctly."""
    # 1. Start task
    payload = {
        "task_type": "recommend",
        "payload": {
            "preferences": {"style": "sporty"},
            "n": 2
        }
    }
    start_resp = client.post("/task/start", json=payload)
    assert start_resp.status_code == 201
    task_id = start_resp.json()["task_id"]

    # 2. Get task
    get_resp = client.get(f"/task/{task_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == "SUCCESS"

    # 3. Delete task
    del_resp = client.delete(f"/task/{task_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "REVOKED"

    # 4. Get task again (should be 404 now)
    get_again = client.get(f"/task/{task_id}")
    assert get_again.status_code == 404

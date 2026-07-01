from __future__ import annotations

import time
import pytest
from week7.backend.services.redis_manager import RedisManager


def test_redis_manager_lifecycle():
    """Verify Redis connection startup and shutdown lifecycle."""
    mgr = RedisManager()
    mgr.connect()
    assert mgr.client is not None
    assert mgr.ping() is True
    mgr.disconnect()
    assert mgr.client is None


def test_redis_caching():
    """Verify general cache set, get, and delete operations."""
    mgr = RedisManager()
    mgr.connect()
    
    # 1. Set cache
    success = mgr.cache_set("test_key", {"data": "hello_world"}, ttl=60)
    assert success is True
    
    # 2. Get cache
    val = mgr.cache_get("test_key")
    assert val == {"data": "hello_world"}
    
    # 3. Delete cache
    deleted = mgr.cache_delete("test_key")
    assert deleted is True
    
    # 4. Get non-existent
    assert mgr.cache_get("test_key") is None
    mgr.disconnect()


def test_redis_sessions():
    """Verify user session storage, retrieval, and invalidation."""
    mgr = RedisManager()
    mgr.connect()
    
    session_data = {"user_id": "usr_123", "role": "designer"}
    assert mgr.session_set("sess_abc", session_data, ttl=60) is True
    assert mgr.session_get("sess_abc") == session_data
    assert mgr.session_delete("sess_abc") is True
    assert mgr.session_get("sess_abc") is None
    mgr.disconnect()


def test_redis_temp_outputs():
    """Verify temporary output storage and retrieval."""
    mgr = RedisManager()
    mgr.connect()
    
    output_meta = {"filename": "out_001.png", "params": {"prompt": "red dress"}}
    assert mgr.store_temp_output("temp_001", output_meta, ttl=60) is True
    assert mgr.get_temp_output("temp_001") == output_meta
    mgr.disconnect()


def test_redis_queuing():
    """Verify FIFO queue enqueuing and dequeuing operations."""
    mgr = RedisManager()
    mgr.connect()
    
    # Enqueue items
    mgr.enqueue("task_queue", {"task_id": 1})
    mgr.enqueue("task_queue", {"task_id": 2})
    
    # Dequeue items
    item1 = mgr.dequeue("task_queue", timeout=1)
    item2 = mgr.dequeue("task_queue", timeout=1)
    
    assert item1 == {"task_id": 1}
    assert item2 == {"task_id": 2}
    
    # Dequeue on empty queue (with timeout)
    empty_item = mgr.dequeue("task_queue", timeout=1)
    assert empty_item is None
    mgr.disconnect()


def test_redis_pubsub():
    """Verify pub/sub messaging channel communication."""
    mgr = RedisManager()
    mgr.connect()
    
    pubsub = mgr.subscribe("alerts")
    time.sleep(0.1)  # Allow subscription to register
    
    # Publish message
    sub_count = mgr.publish("alerts", {"event": "model_loaded"})
    assert sub_count > 0
    
    # Retrieve message
    msg = pubsub.get_message(timeout=1.0)
    assert msg is not None
    assert msg["channel"] == "alerts"
    
    import json
    data = json.loads(msg["data"])
    assert data == {"event": "model_loaded"}
    
    pubsub.unsubscribe("alerts")
    mgr.disconnect()

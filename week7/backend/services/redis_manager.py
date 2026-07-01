from __future__ import annotations

import json
import logging
import queue
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
import redis
from redis.exceptions import ConnectionError, TimeoutError

from week7.backend.configs.config import get_settings

logger = logging.getLogger("redis_manager")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class MockPubSub:
    """Mock implementation of redis.client.PubSub for in-memory fallback."""

    def __init__(self, manager: MockRedisClient) -> None:
        self.manager = manager
        self.channels: List[str] = []
        self.queue: queue.Queue = queue.Queue()

    def subscribe(self, *args, **kwargs) -> None:
        for channel in args:
            if channel not in self.channels:
                self.channels.append(channel)
                self.manager._register_subscriber(channel, self.queue)
        for channel in kwargs:
            if channel not in self.channels:
                self.channels.append(channel)
                self.manager._register_subscriber(channel, self.queue)

    def unsubscribe(self, *args) -> None:
        for channel in args:
            if channel in self.channels:
                self.channels.remove(channel)
                self.manager._unregister_subscriber(channel, self.queue)

    def get_message(self, ignore_subscribe_messages: bool = True, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def listen(self) -> Any:
        while True:
            msg = self.get_message(timeout=1.0)
            if msg:
                yield msg


class MockRedisClient:
    """Thread-safe, in-memory Mock Redis Client for local development/testing fallback."""

    def __init__(self) -> None:
        self._store: Dict[str, bytes] = {}
        self._expires: Dict[str, float] = {}
        self._lists: Dict[str, List[bytes]] = {}
        self._subscribers: Dict[str, List[queue.Queue]] = {}
        self._lock = threading.Lock()

    def _is_expired(self, key: str) -> bool:
        if key in self._expires:
            if time.time() > self._expires[key]:
                self._delete_key(key)
                return True
        return False

    def _delete_key(self, key: str) -> None:
        self._store.pop(key, None)
        self._expires.pop(key, None)
        self._lists.pop(key, None)

    def _register_subscriber(self, channel: str, q: queue.Queue) -> None:
        with self._lock:
            if channel not in self._subscribers:
                self._subscribers[channel] = []
            self._subscribers[channel].append(q)

    def _unregister_subscriber(self, channel: str, q: queue.Queue) -> None:
        with self._lock:
            if channel in self._subscribers:
                if q in self._subscribers[channel]:
                    self._subscribers[channel].remove(q)

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def get(self, name: str) -> Optional[bytes]:
        with self._lock:
            if self._is_expired(name):
                return None
            return self._store.get(name)

    def set(self, name: str, value: Union[str, bytes, int, float], ex: Optional[int] = None) -> bool:
        val_bytes = value if isinstance(value, bytes) else str(value).encode("utf-8")
        with self._lock:
            self._store[name] = val_bytes
            if ex is not None:
                self._expires[name] = time.time() + ex
            else:
                self._expires.pop(name, None)
        return True

    def delete(self, *names: str) -> int:
        count = 0
        with self._lock:
            for name in names:
                if name in self._store or name in self._lists:
                    self._delete_key(name)
                    count += 1
        return count

    def rpush(self, name: str, *values: Union[str, bytes]) -> int:
        with self._lock:
            if name not in self._lists:
                self._lists[name] = []
            for val in values:
                val_bytes = val if isinstance(val, bytes) else str(val).encode("utf-8")
                self._lists[name].append(val_bytes)
            return len(self._lists[name])

    def lpop(self, name: str) -> Optional[bytes]:
        with self._lock:
            if name not in self._lists or not self._lists[name]:
                return None
            return self._lists[name].pop(0)

    def blpop(self, keys: Union[str, List[str]], timeout: int = 0) -> Optional[Tuple[str, bytes]]:
        key_list = [keys] if isinstance(keys, str) else keys
        start_time = time.time()
        while True:
            with self._lock:
                for k in key_list:
                    if k in self._lists and self._lists[k]:
                        val = self._lists[k].pop(0)
                        return k, val
            if timeout > 0 and (time.time() - start_time) >= timeout:
                return None
            time.sleep(0.05)

    def publish(self, channel: str, message: Union[str, bytes]) -> int:
        msg_bytes = message if isinstance(message, bytes) else str(message).encode("utf-8")
        count = 0
        with self._lock:
            subscribers = list(self._subscribers.get(channel, []))
        
        msg_payload_data = msg_bytes.decode("utf-8")
        msg_payload = {
            "type": "message",
            "channel": channel,
            "data": msg_payload_data
        }
        for q in subscribers:
            try:
                q.put(msg_payload)
                count += 1
            except Exception:
                pass
        return count

    def pubsub(self) -> MockPubSub:
        return MockPubSub(self)


class RedisManager:
    """
    Production-ready Redis manager supporting connection management, caching, 
    pub/sub messaging, and FIFO queuing with thread-safe in-memory fallback.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client: Union[redis.Redis, MockRedisClient, None] = None
        self.is_mock: bool = True

    def connect(self) -> None:
        """Establish connection to Redis. Falls back to mock client if connection fails or is disabled."""
        if not self.settings.redis.enabled or self.settings.model.global_mock:
            logger.info("Redis is disabled or global mock is enabled. Initializing in-memory Mock Redis Client.")
            self.client = MockRedisClient()
            self.is_mock = True
            return

        try:
            logger.info(f"Connecting to Redis at {self.settings.redis.host}:{self.settings.redis.port}...")
            self.client = redis.Redis(
                host=self.settings.redis.host,
                port=self.settings.redis.port,
                db=self.settings.redis.db,
                password=self.settings.redis.password,
                socket_timeout=2.0,
                socket_connect_timeout=2.0
            )
            # Verify connection
            self.client.ping()
            self.is_mock = False
            logger.info("Successfully connected to Redis server.")
        except (ConnectionError, TimeoutError, Exception) as exc:
            from week7.backend.logging_config import log_redis_failure
            log_redis_failure("connect", str(exc))
            if self.settings.redis.mock_fallback:
                logger.warning(f"Could not connect to Redis server: {str(exc)}. Falling back to in-memory Mock Redis Client.")
                self.client = MockRedisClient()
                self.is_mock = True
            else:
                logger.error(f"Failed to connect to Redis and mock fallback is disabled: {str(exc)}")
                raise exc

    def ping(self) -> bool:
        """Ping the active Redis client to verify the connection status."""
        if self.client is None:
            self.connect()
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def disconnect(self) -> None:
        """Close connection to Redis."""
        if self.client is not None:
            try:
                self.client.close()
                logger.info("Disconnected from Redis.")
            except Exception as exc:
                logger.warning(f"Error disconnecting from Redis: {str(exc)}")
            finally:
                self.client = None
                self.is_mock = True

    # ── Generic Caching ────────────────────────────────────────────────────────

    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache with optional TTL (seconds). Serializes objects to JSON."""
        if self.client is None:
            self.connect()
        try:
            val_str = json.dumps(value)
            return self.client.set(f"cache:{key}", val_str, ex=ttl)
        except Exception as exc:
            from week7.backend.logging_config import log_redis_failure
            log_redis_failure(f"cache_set:{key}", str(exc))
            logger.error(f"Error writing to cache for key '{key}': {str(exc)}")
            return False

    def cache_get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache. Deserializes JSON string back to Python object."""
        if self.client is None:
            self.connect()
        try:
            data = self.client.get(f"cache:{key}")
            if data is None:
                return None
            return json.loads(data.decode("utf-8"))
        except Exception as exc:
            logger.error(f"Error reading cache for key '{key}': {str(exc)}")
            return None

    def cache_delete(self, key: str) -> bool:
        """Remove key-value pair from cache."""
        if self.client is None:
            self.connect()
        try:
            return bool(self.client.delete(f"cache:{key}"))
        except Exception as exc:
            logger.error(f"Error deleting cache key '{key}': {str(exc)}")
            return False

    # ── User Sessions ──────────────────────────────────────────────────────────

    def session_set(self, session_id: str, data: Dict[str, Any], ttl: Optional[int] = 3600) -> bool:
        """Store user session data. Default TTL is 1 hour."""
        if self.client is None:
            self.connect()
        try:
            val_str = json.dumps(data)
            return self.client.set(f"session:{session_id}", val_str, ex=ttl)
        except Exception as exc:
            logger.error(f"Error setting session '{session_id}': {str(exc)}")
            return False

    def session_get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user session data."""
        if self.client is None:
            self.connect()
        try:
            data = self.client.get(f"session:{session_id}")
            if data is None:
                return None
            return json.loads(data.decode("utf-8"))
        except Exception as exc:
            logger.error(f"Error getting session '{session_id}': {str(exc)}")
            return None

    def session_delete(self, session_id: str) -> bool:
        """Invalidate a user session."""
        if self.client is None:
            self.connect()
        try:
            return bool(self.client.delete(f"session:{session_id}"))
        except Exception as exc:
            logger.error(f"Error deleting session '{session_id}': {str(exc)}")
            return False

    # ── Temporary Outputs ──────────────────────────────────────────────────────

    def store_temp_output(self, key: str, data: Any, ttl: Optional[int] = 600) -> bool:
        """Store temporary outputs/images metadata. Default TTL is 10 minutes."""
        if self.client is None:
            self.connect()
        try:
            val_str = json.dumps(data)
            return self.client.set(f"temp:{key}", val_str, ex=ttl)
        except Exception as exc:
            logger.error(f"Error storing temp output '{key}': {str(exc)}")
            return False

    def get_temp_output(self, key: str) -> Optional[Any]:
        """Retrieve temporary output metadata."""
        if self.client is None:
            self.connect()
        try:
            data = self.client.get(f"temp:{key}")
            if data is None:
                return None
            return json.loads(data.decode("utf-8"))
        except Exception as exc:
            logger.error(f"Error getting temp output '{key}': {str(exc)}")
            return None

    # ── FIFO Queueing ──────────────────────────────────────────────────────────

    def enqueue(self, queue_name: str, item: Any) -> int:
        """Push an item onto a FIFO queue (Right Push)."""
        if self.client is None:
            self.connect()
        try:
            val_str = json.dumps(item)
            return self.client.rpush(f"queue:{queue_name}", val_str)
        except Exception as exc:
            logger.error(f"Error enqueuing item into '{queue_name}': {str(exc)}")
            raise exc

    def dequeue(self, queue_name: str, timeout: int = 0) -> Optional[Any]:
        """Pop an item from a FIFO queue (Blocking Left Pop)."""
        if self.client is None:
            self.connect()
        try:
            # blpop returns (key, value) tuple
            res = self.client.blpop(f"queue:{queue_name}", timeout=timeout)
            if res is None:
                return None
            _, val_bytes = res
            return json.loads(val_bytes.decode("utf-8"))
        except Exception as exc:
            logger.error(f"Error dequeuing item from '{queue_name}': {str(exc)}")
            return None

    # ── Pub/Sub Messaging ──────────────────────────────────────────────────────

    def publish(self, channel: str, message: Any) -> int:
        """Publish a message to a channel."""
        if self.client is None:
            self.connect()
        try:
            val_str = json.dumps(message)
            return self.client.publish(channel, val_str)
        except Exception as exc:
            logger.error(f"Error publishing message to channel '{channel}': {str(exc)}")
            return 0

    def subscribe(self, channel: str) -> Any:
        """Subscribe to a channel and return a pubsub object."""
        if self.client is None:
            self.connect()
        try:
            pubsub = self.client.pubsub()
            pubsub.subscribe(channel)
            return pubsub
        except Exception as exc:
            logger.error(f"Error subscribing to channel '{channel}': {str(exc)}")
            raise exc

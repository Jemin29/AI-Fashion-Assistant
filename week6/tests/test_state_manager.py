"""
Week 6 Test Suite — State Manager.

Verifies session lifecycle, state mutations (model, LoRA, page, preferences),
temporary output tracking, session eviction, statistics aggregation,
snapshot persistence, and health checks.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from week6.services.state_manager import StateManager, SessionState


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mgr() -> StateManager:
    """Return a fresh StateManager in mock_mode (relaxed validation)."""
    return StateManager(mock_mode=True, session_ttl=3600.0, max_sessions=50)


@pytest.fixture
def session_id(mgr: StateManager) -> str:
    """Create a session and return its ID."""
    res = mgr.get_or_create_session("sess-001", user_id="test_user")
    assert res.is_ok
    return "sess-001"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Session Lifecycle
# ══════════════════════════════════════════════════════════════════════════════

def test_create_new_session(mgr: StateManager) -> None:
    """Creating a session returns a valid SessionState."""
    res = mgr.get_or_create_session("new-session", user_id="alice")
    assert res.is_ok
    s = res.data
    assert isinstance(s, SessionState)
    assert s.session_id == "new-session"
    assert s.user_id == "alice"
    assert s.active_model == "SDXL v1.0 (Mock)"
    assert s.selected_lora == ""
    assert s.active_page == "🏠 Home Dashboard"
    assert res.meta["created"] is True


def test_retrieve_existing_session(mgr: StateManager) -> None:
    """Retrieving an existing session returns the same object (not a new one)."""
    mgr.get_or_create_session("existing-s")
    res = mgr.get_or_create_session("existing-s")
    assert res.is_ok
    assert res.meta["created"] is False


def test_get_session_not_found(mgr: StateManager) -> None:
    """Getting a missing session returns a validation failure."""
    res = mgr.get_session("does-not-exist")
    assert not res.is_ok
    assert "session_id" in res.error


def test_delete_session(mgr: StateManager, session_id: str) -> None:
    """Deleting a session succeeds and removes it from the store."""
    res = mgr.delete_session(session_id)
    assert res.is_ok
    assert res.data["session_id"] == session_id

    # Subsequent get should fail
    get_res = mgr.get_session(session_id)
    assert not get_res.is_ok


def test_delete_nonexistent_session(mgr: StateManager) -> None:
    """Deleting a non-existent session returns a validation failure."""
    res = mgr.delete_session("ghost-session")
    assert not res.is_ok


def test_auto_generate_session_id(mgr: StateManager) -> None:
    """Calling get_or_create_session without an ID generates a UUID."""
    res = mgr.get_or_create_session()
    assert res.is_ok
    assert len(res.data.session_id) == 36  # UUID4 length
    assert res.meta["created"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 2. Active Model
# ══════════════════════════════════════════════════════════════════════════════

def test_set_active_model(mgr: StateManager, session_id: str) -> None:
    """Setting an active model updates the session."""
    res = mgr.set_active_model(session_id, "SDXL Lightning (Mock)")
    assert res.is_ok
    assert res.data == "SDXL Lightning (Mock)"

    s_res = mgr.get_session(session_id)
    assert s_res.data.active_model == "SDXL Lightning (Mock)"


def test_set_active_model_invalid_session(mgr: StateManager) -> None:
    """Setting model on unknown session fails."""
    res = mgr.set_active_model("ghost", "SDXL v1.0 (Mock)")
    assert not res.is_ok


def test_set_active_model_empty_string(mgr: StateManager, session_id: str) -> None:
    """Empty model string is rejected."""
    res = mgr.set_active_model(session_id, "")
    assert not res.is_ok
    assert "model" in res.error


# ══════════════════════════════════════════════════════════════════════════════
# 3. Selected LoRA
# ══════════════════════════════════════════════════════════════════════════════

def test_set_selected_lora(mgr: StateManager, session_id: str) -> None:
    """Setting a LoRA adapter updates the session."""
    res = mgr.set_selected_lora(session_id, "nike_streetwear_v2", weight=0.80)
    assert res.is_ok
    assert res.data["lora"] == "nike_streetwear_v2"
    assert abs(res.data["weight"] - 0.80) < 1e-6

    s = mgr.get_session(session_id).data
    assert s.selected_lora == "nike_streetwear_v2"
    assert abs(s.lora_weight - 0.80) < 1e-6


def test_clear_selected_lora(mgr: StateManager, session_id: str) -> None:
    """Passing empty string deselects the LoRA."""
    mgr.set_selected_lora(session_id, "gucci_luxury_v1", weight=0.6)
    res = mgr.set_selected_lora(session_id, "", weight=0.0)
    assert res.is_ok
    assert res.data["lora"] == ""


def test_set_lora_invalid_weight(mgr: StateManager, session_id: str) -> None:
    """LoRA weight outside [0.0, 1.0] is rejected."""
    res = mgr.set_selected_lora(session_id, "gucci_luxury_v1", weight=2.5)
    assert not res.is_ok
    assert "weight" in res.error


def test_set_lora_non_numeric_weight(mgr: StateManager, session_id: str) -> None:
    """Non-numeric LoRA weight is rejected."""
    res = mgr.set_selected_lora(session_id, "gucci_luxury_v1", weight="high")  # type: ignore[arg-type]
    assert not res.is_ok


# ══════════════════════════════════════════════════════════════════════════════
# 4. Active Page
# ══════════════════════════════════════════════════════════════════════════════

def test_set_active_page(mgr: StateManager, session_id: str) -> None:
    """Setting active page persists to session."""
    res = mgr.set_active_page(session_id, "🎨 Text-to-Fashion")
    assert res.is_ok
    assert res.data == "🎨 Text-to-Fashion"

    s = mgr.get_session(session_id).data
    assert s.active_page == "🎨 Text-to-Fashion"


def test_set_active_page_invalid_session(mgr: StateManager) -> None:
    """Setting page on an unknown session fails."""
    res = mgr.set_active_page("ghost", "🖼️ Gallery")
    assert not res.is_ok


def test_set_active_page_empty(mgr: StateManager, session_id: str) -> None:
    """Empty page name is rejected."""
    res = mgr.set_active_page(session_id, "")
    assert not res.is_ok


# ══════════════════════════════════════════════════════════════════════════════
# 5. Generation Context & Chat Turns
# ══════════════════════════════════════════════════════════════════════════════

def test_update_generation_context(mgr: StateManager, session_id: str) -> None:
    """Recording a generation context increments counter and stores prompt."""
    res = mgr.update_generation_context(
        session_id,
        prompt="silk evening gown, haute couture",
        image_path="/outputs/generated/img001.png",
    )
    assert res.is_ok
    s = mgr.get_session(session_id).data
    assert s.last_prompt == "silk evening gown, haute couture"
    assert s.last_image_path == "/outputs/generated/img001.png"
    assert s.generation_count == 1


def test_increment_chat_turns(mgr: StateManager, session_id: str) -> None:
    """Chat turn counter increments on each call."""
    for i in range(3):
        res = mgr.increment_chat_turns(session_id)
        assert res.is_ok
        assert res.data == i + 1

    s = mgr.get_session(session_id).data
    assert s.chat_turns == 3


# ══════════════════════════════════════════════════════════════════════════════
# 6. User Preferences
# ══════════════════════════════════════════════════════════════════════════════

def test_set_preference(mgr: StateManager, session_id: str) -> None:
    """Arbitrary preferences are stored and retrievable."""
    res = mgr.set_preference(session_id, "theme", "dark_mode")
    assert res.is_ok
    assert res.data == {"theme": "dark_mode"}

    s = mgr.get_session(session_id).data
    assert s.preferences["theme"] == "dark_mode"


def test_set_preference_empty_key(mgr: StateManager, session_id: str) -> None:
    """Empty preference key is rejected."""
    res = mgr.set_preference(session_id, "", "value")
    assert not res.is_ok


# ══════════════════════════════════════════════════════════════════════════════
# 7. Temporary Output Management
# ══════════════════════════════════════════════════════════════════════════════

def test_add_and_get_temporary_outputs(mgr: StateManager, session_id: str, tmp_path: Path) -> None:
    """Temporary output paths are registered and retrievable."""
    f1 = tmp_path / "preview_01.png"
    f1.write_bytes(b"\x89PNG")
    f2 = tmp_path / "export_batch.zip"
    f2.write_bytes(b"PK\x03\x04")

    mgr.add_temporary_output(session_id, str(f1))
    mgr.add_temporary_output(session_id, str(f2))

    res = mgr.get_temporary_outputs(session_id)
    assert res.is_ok
    assert len(res.data) == 2
    assert str(f1) in res.data
    assert str(f2) in res.data


def test_add_temp_output_no_duplicates(mgr: StateManager, session_id: str, tmp_path: Path) -> None:
    """Adding the same path twice only registers it once."""
    f = tmp_path / "dup.png"
    f.write_bytes(b"IMG")

    mgr.add_temporary_output(session_id, str(f))
    mgr.add_temporary_output(session_id, str(f))

    res = mgr.get_temporary_outputs(session_id)
    assert len(res.data) == 1


def test_clear_temporary_outputs_deletes_files(
    mgr: StateManager, session_id: str, tmp_path: Path
) -> None:
    """Clearing temp outputs deletes the actual files from disk."""
    f = tmp_path / "temp_render.png"
    f.write_bytes(b"\x89PNG")
    mgr.add_temporary_output(session_id, str(f))

    res = mgr.clear_temporary_outputs(session_id)
    assert res.is_ok
    assert res.data["files_cleaned"] == 1
    assert res.data["files_failed"] == 0
    assert not f.exists()

    # Session should still exist but with empty temp list
    s = mgr.get_session(session_id).data
    assert len(s.temporary_outputs) == 0


def test_clear_temp_outputs_nonexistent_files(
    mgr: StateManager, session_id: str
) -> None:
    """Clearing temp paths that do not exist on disk is counted as cleaned (not failed)."""
    mgr.add_temporary_output(session_id, "/nonexistent/path/phantom.png")

    res = mgr.clear_temporary_outputs(session_id)
    assert res.is_ok
    # File does not exist → counted as cleaned (no error raised)
    assert res.data["files_failed"] == 0


def test_delete_session_cleans_temp_files(
    mgr: StateManager, tmp_path: Path
) -> None:
    """Deleting a session also cleans up its registered temporary files."""
    mgr.get_or_create_session("cleanup-sess")
    f = tmp_path / "to_delete.png"
    f.write_bytes(b"PNG")
    mgr.add_temporary_output("cleanup-sess", str(f))

    res = mgr.delete_session("cleanup-sess")
    assert res.is_ok
    assert res.data["files_cleaned"] == 1
    assert not f.exists()


def test_add_temp_output_empty_path(mgr: StateManager, session_id: str) -> None:
    """Empty file path is rejected."""
    res = mgr.add_temporary_output(session_id, "")
    assert not res.is_ok


# ══════════════════════════════════════════════════════════════════════════════
# 8. Session Listing & Statistics
# ══════════════════════════════════════════════════════════════════════════════

def test_list_sessions(mgr: StateManager) -> None:
    """list_sessions returns all active sessions sorted by creation time."""
    mgr.get_or_create_session("s-a")
    mgr.get_or_create_session("s-b")
    mgr.get_or_create_session("s-c")

    res = mgr.list_sessions()
    assert res.is_ok
    assert res.meta["total_sessions"] == 3
    ids = [d["session_id"] for d in res.data]
    assert "s-a" in ids and "s-b" in ids and "s-c" in ids


def test_get_stats(mgr: StateManager) -> None:
    """get_stats aggregates counts across all sessions."""
    mgr.get_or_create_session("stat-1")
    mgr.get_or_create_session("stat-2")
    mgr.set_active_model("stat-1", "SDXL Lightning (Mock)")
    mgr.update_generation_context("stat-1", prompt="test")
    mgr.update_generation_context("stat-2", prompt="test2")
    mgr.increment_chat_turns("stat-1")

    res = mgr.get_stats()
    assert res.is_ok
    stats = res.data
    assert stats["total_sessions"] == 2
    assert stats["total_generations"] == 2
    assert stats["total_chat_turns"] == 1
    assert "SDXL Lightning (Mock)" in stats["model_distribution"]


# ══════════════════════════════════════════════════════════════════════════════
# 9. Session Eviction
# ══════════════════════════════════════════════════════════════════════════════

def test_evict_stale_sessions(mgr: StateManager) -> None:
    """Sessions idle longer than TTL are evicted."""
    # Create a manager with very short TTL
    short_ttl_mgr = StateManager(mock_mode=True, session_ttl=0.01)
    short_ttl_mgr.get_or_create_session("stale-1")
    short_ttl_mgr.get_or_create_session("stale-2")

    # Wait slightly longer than TTL
    time.sleep(0.05)

    res = short_ttl_mgr.evict_stale_sessions()
    assert res.is_ok
    assert res.data == 2

    # Sessions should be gone
    assert not short_ttl_mgr.get_session("stale-1").is_ok
    assert not short_ttl_mgr.get_session("stale-2").is_ok


def test_max_sessions_eviction(mgr: StateManager) -> None:
    """Exceeding max_sessions triggers LRU eviction of oldest sessions."""
    small_mgr = StateManager(mock_mode=True, max_sessions=3)
    for i in range(3):
        small_mgr.get_or_create_session(f"cap-{i}")

    # One more session should trigger eviction of the oldest
    small_mgr.get_or_create_session("cap-overflow")

    # Total sessions in manager should be ≤ max_sessions
    list_res = small_mgr.list_sessions()
    assert list_res.data is not None
    assert len(list_res.data) <= 3


# ══════════════════════════════════════════════════════════════════════════════
# 10. Snapshot Persistence
# ══════════════════════════════════════════════════════════════════════════════

def test_dump_and_load_snapshot(mgr: StateManager, tmp_path: Path) -> None:
    """Sessions survive a dump → load round-trip."""
    snap_file = tmp_path / "snap.json"
    mgr.get_or_create_session("persist-1", user_id="alice")
    mgr.set_active_model("persist-1", "SDXL Lightning (Mock)")
    mgr.set_selected_lora("persist-1", "nike_streetwear_v2", weight=0.6)
    mgr.set_active_page("persist-1", "🎨 Text-to-Fashion")

    dump_res = mgr.dump_snapshot(snap_file)
    assert dump_res.is_ok
    assert snap_file.exists()

    # Load into a fresh manager
    fresh_mgr = StateManager(mock_mode=True)
    load_res = fresh_mgr.load_snapshot(snap_file)
    assert load_res.is_ok
    assert load_res.data == 1

    s = fresh_mgr.get_session("persist-1").data
    assert s.user_id == "alice"
    assert s.active_model == "SDXL Lightning (Mock)"
    assert s.selected_lora == "nike_streetwear_v2"
    assert s.active_page == "🎨 Text-to-Fashion"


def test_load_snapshot_missing_file(mgr: StateManager, tmp_path: Path) -> None:
    """Loading from a non-existent snapshot returns a failure result."""
    res = mgr.load_snapshot(tmp_path / "nonexistent.json")
    assert not res.is_ok
    assert "SNAPSHOT_NOT_FOUND" in res.error_code


# ══════════════════════════════════════════════════════════════════════════════
# 11. SessionState Serialization
# ══════════════════════════════════════════════════════════════════════════════

def test_session_state_to_dict(mgr: StateManager, session_id: str) -> None:
    """SessionState.to_dict() produces JSON-serializable output."""
    s = mgr.get_session(session_id).data
    d = s.to_dict()

    # Must be JSON-serializable
    dumped = json.dumps(d)
    assert "session_id" in dumped

    assert d["session_id"] == session_id
    assert isinstance(d["lora_weight"], float)
    assert isinstance(d["generation_count"], int)
    assert isinstance(d["temporary_outputs"], list)


def test_session_state_from_dict_roundtrip() -> None:
    """SessionState serialization roundtrip preserves all fields."""
    original = SessionState(
        session_id="rt-test",
        user_id="roundtrip_user",
        active_model="SDXL v1.0 Turbo (Mock)",
        selected_lora="gucci_luxury_v1",
        lora_weight=0.55,
        active_page="🖼️ Gallery",
        generation_count=7,
        chat_turns=3,
        temporary_outputs=["/a/b/c.png"],
        last_prompt="velvet gown editorial",
        preferences={"lang": "en"},
    )
    restored = SessionState.from_dict(original.to_dict())
    assert restored.session_id == original.session_id
    assert restored.active_model == original.active_model
    assert restored.selected_lora == original.selected_lora
    assert abs(restored.lora_weight - original.lora_weight) < 1e-6
    assert restored.generation_count == original.generation_count
    assert restored.preferences == original.preferences


# ══════════════════════════════════════════════════════════════════════════════
# 12. Health Check
# ══════════════════════════════════════════════════════════════════════════════

def test_health_check(mgr: StateManager) -> None:
    """health_check returns status=ok and key metrics."""
    hc = mgr.health_check()
    assert hc["status"] == "ok"
    assert "readable/writable" in hc["message"]
    assert "total_sessions" in hc
    assert "max_sessions" in hc
    assert "session_ttl_s" in hc


# ══════════════════════════════════════════════════════════════════════════════
# 13. Thread Safety
# ══════════════════════════════════════════════════════════════════════════════

def test_concurrent_session_creation(mgr: StateManager) -> None:
    """Concurrent session creates from multiple threads don't corrupt state."""
    errors: list[Exception] = []

    def create(idx: int) -> None:
        try:
            mgr.get_or_create_session(f"thread-sess-{idx}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=create, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    list_res = mgr.list_sessions()
    assert list_res.meta["total_sessions"] == 20


def test_concurrent_state_mutations(mgr: StateManager, session_id: str) -> None:
    """Concurrent increments to generation_count produce a consistent final count."""
    errors: list[Exception] = []

    def increment(n: int) -> None:
        try:
            for _ in range(n):
                mgr.update_generation_context(session_id, prompt="concurrent test")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=increment, args=(10,)) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    s = mgr.get_session(session_id).data
    assert s.generation_count == 50

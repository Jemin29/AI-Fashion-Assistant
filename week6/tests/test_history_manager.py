"""
Week 6 Test Suite — Generation History Manager.
Verifies persistence, search, pagination, stats, deletion, and exports.
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest

from week6.services.history_manager import HistoryManager, HistoryEntry


@pytest.fixture
def temp_history_file(tmp_path) -> Path:
    """Fixture returning a temp path for history file."""
    return tmp_path / "temp_history.jsonl"


def test_history_manager_basic_flow(temp_history_file) -> None:
    """Test standard recording, loading, and listing functions."""
    mgr = HistoryManager(history_file=temp_history_file)
    
    # Assert starts empty
    entries, total = mgr.list_entries(page=1, page_size=10)
    assert total == 0
    assert len(entries) == 0

    # Record first entry
    entry1 = mgr.record(
        prompt="A chic silk gown in neutral ivory",
        image_path="/path/to/img1.png",
        negative_prompt="blurry",
        model="SDXL v1.0",
        style="Minimalist Studio",
        mode="mock",
        seed=101,
        steps=25,
        guidance_scale=8.0,
        latency_ms=150.5,
        tags=["silk", "gown", "ivory"]
    )
    
    assert entry1.id is not None
    assert entry1.prompt == "A chic silk gown in neutral ivory"
    assert entry1.seed == 101
    assert entry1.image_exists is False  # image path doesn't actually exist on disk

    # Record second entry
    entry2 = mgr.record(
        prompt="Streetwear jacket with bold prints",
        image_path="/path/to/img2.png",
        model="SDXL v1.0",
        style="Streetwear Editorial",
        mode="mock",
        seed=202,
        steps=30,
        guidance_scale=7.5,
        latency_ms=250.0,
        tags=["streetwear", "jacket"]
    )

    # Verify count is 2
    entries, total = mgr.list_entries(page=1, page_size=10)
    assert total == 2
    assert len(entries) == 2
    
    # Reload from file and verify it persists
    mgr_reload = HistoryManager(history_file=temp_history_file)
    entries_reload, total_reload = mgr_reload.list_entries(page=1, page_size=10)
    assert total_reload == 2
    
    # Test get_entry
    e = mgr_reload.get_entry(entry1.id)
    assert e is not None
    assert e.prompt == "A chic silk gown in neutral ivory"


def test_history_manager_search_and_filters(temp_history_file) -> None:
    """Test text search and complex field filtering."""
    mgr = HistoryManager(history_file=temp_history_file)

    mgr.record(prompt="Ivory silk dress", style="Minimalist", rating=5)
    mgr.record(prompt="Neon leather jacket", style="Streetwear", rating=3)
    mgr.record(prompt="Blue denim jeans", style="Casual", rating=4)

    # Text search
    entries, total = mgr.search(query="silk")
    assert total == 1
    assert entries[0].prompt == "Ivory silk dress"

    # Search with filters
    entries, total = mgr.search(query="", filters={"rating_min": 4})
    assert total == 2  # Ivory silk dress (5) and Blue denim jeans (4)

    entries, total = mgr.search(query="", filters={"style": "streetwear"})
    assert total == 1
    assert entries[0].prompt == "Neon leather jacket"


def test_history_manager_sorting_and_pagination(temp_history_file, monkeypatch) -> None:
    """Test sorting modes and pagination limits."""
    mgr = HistoryManager(history_file=temp_history_file)

    # Mock time.time to increment dynamically
    t_val = 1000.0
    def mock_time():
        nonlocal t_val
        t_val += 1.0
        return t_val
    monkeypatch.setattr("time.time", mock_time)

    # Insert 5 entries
    for i in range(1, 6):
        mgr.record(prompt=f"Design {i}", latency_ms=10.0 * i, rating=i)

    # Page 1 size 2
    entries, total = mgr.list_entries(page=1, page_size=2, sort="newest")
    assert total == 5
    assert len(entries) == 2
    assert entries[0].prompt == "Design 5"
    assert entries[1].prompt == "Design 4"

    # Page 3 size 2
    entries, total = mgr.list_entries(page=3, page_size=2, sort="newest")
    assert len(entries) == 1
    assert entries[0].prompt == "Design 1"

    # Sort by latency oldest-first (meaning smallest first)
    entries, total = mgr.list_entries(page=1, page_size=5, sort="latency")
    assert entries[0].prompt == "Design 1"
    assert entries[-1].prompt == "Design 5"


def test_history_manager_update_and_delete(temp_history_file) -> None:
    """Test updates and deletion."""
    mgr = HistoryManager(history_file=temp_history_file)

    entry = mgr.record(prompt="Original prompt", rating=0, notes="")
    
    # Update entry
    mgr.update_entry(entry.id, rating=5, notes="Stunning design concept")
    updated = mgr.get_entry(entry.id)
    assert updated.rating == 5
    assert updated.notes == "Stunning design concept"

    # Delete entry
    deleted = mgr.delete_entry(entry.id, delete_file=False)
    assert deleted is True
    assert mgr.get_entry(entry.id) is None

    # Test delete non-existent
    assert mgr.delete_entry("invalid-id") is False


def test_history_manager_bulk_operations(temp_history_file) -> None:
    """Test bulk delete and stats aggregation."""
    mgr = HistoryManager(history_file=temp_history_file)

    e1 = mgr.record(prompt="Design A", model="Model X", latency_ms=100.0)
    e2 = mgr.record(prompt="Design B", model="Model Y", latency_ms=200.0)
    e3 = mgr.record(prompt="Design C", model="Model X", latency_ms=150.0)

    # Bulk delete
    count = mgr.delete_many([e1.id, e2.id], delete_files=False)
    assert count == 2
    assert mgr.get_entry(e1.id) is None
    assert mgr.get_entry(e3.id) is not None

    # Stats
    stats = mgr.get_stats()
    assert stats["total"] == 1
    assert stats["by_model"]["Model X"] == 1


def test_history_manager_prune_and_eviction(temp_history_file) -> None:
    """Test old record pruning and oldest-first eviction."""
    mgr = HistoryManager(history_file=temp_history_file, max_records=3)

    e1 = mgr.record(prompt="Design 1")
    e2 = mgr.record(prompt="Design 2")
    e3 = mgr.record(prompt="Design 3")
    
    # Records count is 3
    assert len(mgr._records) == 3

    # Add 4th record - triggers eviction of oldest (Design 1)
    e4 = mgr.record(prompt="Design 4")
    assert len(mgr._records) == 3
    assert mgr.get_entry(e1.id) is None
    assert mgr.get_entry(e4.id) is not None

    # Prune by TTL
    pruned = mgr.prune_old(ttl_days=-1) # force prune everything
    assert pruned == 3
    assert len(mgr._records) == 0


def test_history_manager_exports(temp_history_file, tmp_path) -> None:
    """Test export utilities."""
    # Temporarily override export directory constants inside history_manager module if needed
    import week6.services.history_manager as hm
    old_exports_dir = hm._EXPORTS_DIR
    hm._EXPORTS_DIR = tmp_path / "exports"
    hm._EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        mgr = HistoryManager(history_file=temp_history_file)
        mgr.record(prompt="Export Design", rating=5, notes="Export check")

        json_path = mgr.export_json()
        assert Path(json_path).exists()
        assert str(json_path).endswith(".json")

        csv_path = mgr.export_csv()
        assert Path(csv_path).exists()
        assert str(csv_path).endswith(".csv")

        md_path = mgr.export_markdown()
        assert Path(md_path).exists()
        assert str(md_path).endswith(".md")
    finally:
        hm._EXPORTS_DIR = old_exports_dir


def test_history_entry_convenience() -> None:
    """Test dataclass properties and conversion methods."""
    entry = HistoryEntry(
        prompt="A very very long prompt text to test the short_prompt truncation property of HistoryEntry",
        rating=4,
        notes="Nice garment outline"
    )
    
    assert len(entry.short_prompt) <= 60
    assert entry.short_prompt.endswith("…")
    
    # Dict conversion
    d = entry.to_dict()
    assert d["rating"] == 4
    assert d["notes"] == "Nice garment outline"

    rebuilt = HistoryEntry.from_dict(d)
    assert rebuilt.notes == "Nice garment outline"
    assert rebuilt.created_dt is not None

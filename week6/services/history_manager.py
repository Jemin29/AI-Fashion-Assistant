"""
Week 6 — Generation History Manager.

Persistent, thread-safe store of every image generation event produced by
the AI Fashion Creative Studio.  Records are written to a single JSON Lines
file (``history.jsonl``) and an in-memory index is maintained for fast
search and filter operations.

Features
--------
- **Append** — record one generation event atomically (prompt, image path,
  model, style, mode, seed, resolution, latency, timestamps, tags)
- **View**    — paginated listing with rich filtering and sorting
- **Search**  — full-text + field-level search across all record fields
- **Delete**  — remove individual records or bulk-delete by filter
- **Export**  — export history to JSON, CSV, or Markdown report
- **Stats**   — aggregated usage statistics (model breakdown, style counts, etc.)
- **Prune**   — automatic eviction of records older than a configurable TTL

Thread safety
-------------
All mutations are protected by a ``threading.Lock``.  The JSON Lines format
guarantees atomic append even if the process is killed mid-write (the last
incomplete line is silently ignored on next load).

File layout
-----------
::

    week6/outputs/history/
    ├── history.jsonl        # append-only record log
    ├── history_index.json   # rebuild-able in-memory index snapshot
    └── exports/
        ├── history_export_<ts>.json
        ├── history_export_<ts>.csv
        └── history_report_<ts>.md

Usage
-----
::

    from week6.services.history_manager import HistoryManager

    mgr = HistoryManager()

    # Record a generation
    entry = mgr.record(
        prompt="elegant silk dress, minimalist studio",
        image_path="/path/to/output.png",
        model="SDXL v1.0 (Mock)",
        style="Minimalist Studio",
        mode="mock",
        seed=42,
        resolution="1024×1024",
        latency_ms=320.5,
        tags=["silk", "dress", "minimalist"],
    )
    print(entry.id)

    # Search
    results = mgr.search("silk dress")

    # Export
    path = mgr.export_json()
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from week6.gradio_app.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

# ── Storage paths ──────────────────────────────────────────────────────────────
_HISTORY_DIR  = Path(__file__).resolve().parent.parent / "outputs" / "history"
_HISTORY_FILE = _HISTORY_DIR / "history.jsonl"
_INDEX_FILE   = _HISTORY_DIR / "history_index.json"
_EXPORTS_DIR  = _HISTORY_DIR / "exports"

_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_HISTORY_RECORDS  = 10_000   # hard cap before oldest-first eviction
DEFAULT_PAGE_SIZE    = 20
DEFAULT_TTL_DAYS     = 90       # prune records older than this by default


# ══════════════════════════════════════════════════════════════════════════════
# Data model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HistoryEntry:
    """A single generation history record.

    Attributes:
        id:            Unique record identifier (UUID4 hex).
        prompt:        The generation prompt text.
        negative_prompt: Optional negative prompt.
        image_path:    Absolute path to the saved image file.
        thumbnail_path:Path to a thumbnail image (may be empty).
        model:         Model name used for generation.
        style:         Style preset name (empty if none).
        mode:          Generation mode: ``"real"`` / ``"mock"`` / ``"error"``.
        service:       Originating service: ``"generation"`` / ``"controlnet"`` / ``"lora"``.
        brand:         LoRA brand name (empty if not applicable).
        conditioning_mode: ControlNet mode (empty if not applicable).
        resolution:    Resolution string e.g. ``"1024×1024"``.
        seed:          Random seed (``None`` if not set).
        steps:         Number of inference steps.
        guidance_scale:CFG guidance scale.
        lora_scale:    LoRA adapter scale (0 if not applicable).
        latency_ms:    End-to-end latency in milliseconds.
        tags:          User-assigned or auto-generated tag list.
        rating:        Optional user rating [1–5] (0 = unrated).
        notes:         Optional free-text user notes.
        created_at:    ISO-8601 UTC timestamp of record creation.
        created_ts:    Unix timestamp (float) for fast sorting/filtering.
        deleted:       Soft-delete flag (hard deletion removes the record entirely).
    """

    id:                str   = field(default_factory=lambda: uuid.uuid4().hex)
    prompt:            str   = ""
    negative_prompt:   str   = ""
    image_path:        str   = ""
    thumbnail_path:    str   = ""
    model:             str   = ""
    style:             str   = ""
    mode:              str   = "mock"
    service:           str   = "generation"
    brand:             str   = ""
    conditioning_mode: str   = ""
    resolution:        str   = ""
    seed:              Optional[int] = None
    steps:             int   = 0
    guidance_scale:    float = 7.5
    lora_scale:        float = 0.0
    latency_ms:        float = 0.0
    tags:              List[str] = field(default_factory=list)
    rating:            int   = 0
    notes:             str   = ""
    created_at:        str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_ts:        float = field(default_factory=time.time)
    deleted:           bool  = False

    # ── Convenience ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HistoryEntry":
        """Deserialise from a dict (unknown keys are ignored gracefully)."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def short_prompt(self) -> str:
        """Prompt truncated to 60 characters for display."""
        return (self.prompt[:57] + "…") if len(self.prompt) > 60 else self.prompt

    @property
    def image_exists(self) -> bool:
        """``True`` if the saved image file is still on disk."""
        return bool(self.image_path) and Path(self.image_path).exists()

    @property
    def created_dt(self) -> datetime:
        """``created_at`` parsed as a ``datetime`` object."""
        try:
            return datetime.fromisoformat(self.created_at)
        except ValueError:
            return datetime.fromtimestamp(self.created_ts, tz=timezone.utc)

    def matches_text(self, query: str) -> bool:
        """Return ``True`` if *query* appears in any searchable text field."""
        q = query.lower()
        haystack = " ".join([
            str(self.prompt or ""),
            str(self.negative_prompt or ""),
            str(self.model or ""),
            str(self.style or ""),
            str(self.brand or ""),
            str(self.conditioning_mode or ""),
            str(self.mode or ""),
            str(self.service or ""),
            str(self.notes or ""),
            " ".join(self.tags),
        ]).lower()
        return q in haystack


    def matches_filter(self, filters: Dict[str, Any]) -> bool:
        """Return ``True`` if this entry satisfies all filter predicates.

        Supported filter keys:

        - ``model``              — exact or partial model name match
        - ``style``              — exact or partial style match
        - ``mode``               — exact mode match (real / mock / error)
        - ``service``            — exact service match
        - ``brand``              — exact brand match
        - ``conditioning_mode``  — exact conditioning mode match
        - ``resolution``         — exact resolution string match
        - ``tag``                — entry must contain this tag
        - ``rating_min``         — minimum rating (inclusive)
        - ``rating_max``         — maximum rating (inclusive)
        - ``date_from``          — unix timestamp lower bound (inclusive)
        - ``date_to``            — unix timestamp upper bound (inclusive)
        - ``has_image``          — bool: image file must exist on disk
        - ``seed``               — exact seed match
        """
        for key, val in filters.items():
            if val is None or val == "" or val == []:
                continue
            if key == "model"              and val.lower() not in self.model.lower():             return False
            if key == "style"              and val.lower() not in self.style.lower():             return False
            if key == "mode"               and val.lower() != self.mode.lower():                   return False
            if key == "service"            and val.lower() != self.service.lower():                return False
            if key == "brand"              and val.lower() not in self.brand.lower():              return False
            if key == "conditioning_mode"  and val.lower() != self.conditioning_mode.lower():      return False
            if key == "resolution"         and val != self.resolution:                             return False
            if key == "tag"                and val.lower() not in [t.lower() for t in self.tags]: return False
            if key == "rating_min"         and self.rating < int(val):                            return False
            if key == "rating_max"         and self.rating > int(val):                            return False
            if key == "date_from"          and self.created_ts < float(val):                      return False
            if key == "date_to"            and self.created_ts > float(val):                      return False
            if key == "has_image"          and bool(val) != self.image_exists:                    return False
            if key == "seed"               and self.seed != int(val):                             return False
        return True


# ── Sort key helpers ───────────────────────────────────────────────────────────

_SORT_KEYS: Dict[str, Callable[[HistoryEntry], Any]] = {
    "newest":     lambda e: -e.created_ts,
    "oldest":     lambda e:  e.created_ts,
    "prompt":     lambda e:  e.prompt.lower(),
    "model":      lambda e:  e.model.lower(),
    "style":      lambda e:  e.style.lower(),
    "rating":     lambda e: -e.rating,
    "latency":    lambda e:  e.latency_ms,
}


# ══════════════════════════════════════════════════════════════════════════════
# HistoryManager
# ══════════════════════════════════════════════════════════════════════════════

class HistoryManager:
    """
    Thread-safe persistent history store for AI Fashion generation events.

    All write operations are protected by a ``threading.RLock``.  Records are
    durably appended to a JSON Lines file so no history is lost on crash.

    Args:
        history_file: Path to the ``.jsonl`` store (default: ``outputs/history/history.jsonl``).
        max_records:  Hard capacity cap.  When exceeded, oldest records are evicted.
        auto_load:    If ``True`` (default), load existing records on init.
    """

    def __init__(
        self,
        history_file: Optional[Path] = None,
        max_records: int = MAX_HISTORY_RECORDS,
        auto_load: bool = True,
    ) -> None:
        self._file       = Path(history_file) if history_file else _HISTORY_FILE
        self._file.parent.mkdir(parents=True, exist_ok=True)

        self._max        = max_records
        self._lock       = threading.RLock()
        self._records:   Dict[str, HistoryEntry] = {}   # id → entry (insertion order preserved)
        self._dirty      = False

        if auto_load:
            self._load()

        logger.info(
            "HistoryManager: loaded %d records from %s", len(self._records), self._file
        )

    # ── Internal I/O ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load all non-deleted records from the JSON Lines file."""
        if not self._file.exists():
            return
        with self._lock:
            self._records.clear()
            corrupt = 0
            with open(self._file, "r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        entry = HistoryEntry.from_dict(d)
                        if not entry.deleted:
                            self._records[entry.id] = entry
                    except (json.JSONDecodeError, TypeError, ValueError):
                        corrupt += 1
                        logger.warning("HistoryManager: corrupt record at line %d — skipped", lineno)
            if corrupt:
                logger.warning("HistoryManager: %d corrupt lines ignored", corrupt)

    def _append_to_file(self, entry: HistoryEntry) -> None:
        """Atomically append a single record to the JSON Lines file."""
        line = json.dumps(entry.to_dict(), ensure_ascii=False)
        with open(self._file, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _rewrite_file(self) -> None:
        """Rewrite the JSONL file from the current in-memory state.

        Used after bulk deletes or prune operations to compact the file.
        """
        tmp = self._file.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for entry in self._records.values():
                fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(self._file)

    def _evict_oldest(self, n: int) -> None:
        """Remove the *n* oldest records from memory (does NOT rewrite file immediately)."""
        sorted_ids = sorted(self._records, key=lambda i: self._records[i].created_ts)
        for rid in sorted_ids[:n]:
            del self._records[rid]

    # ── Record ─────────────────────────────────────────────────────────────────

    def record(
        self,
        prompt: str,
        image_path: str = "",
        *,
        negative_prompt: str = "",
        model: str = "SDXL v1.0 (Mock)",
        style: str = "",
        mode: str = "mock",
        service: str = "generation",
        brand: str = "",
        conditioning_mode: str = "",
        resolution: str = "",
        seed: Optional[int] = None,
        steps: int = 0,
        guidance_scale: float = 7.5,
        lora_scale: float = 0.0,
        latency_ms: float = 0.0,
        tags: Optional[List[str]] = None,
        rating: int = 0,
        notes: str = "",
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> HistoryEntry:
        """Append a new generation event to the history store.

        Args:
            prompt:            The text prompt used for generation.
            image_path:        Absolute path to the saved output image.
            negative_prompt:   Negative conditioning text (optional).
            model:             Model/pipeline name.
            style:             Style preset name.
            mode:              Run mode: ``"real"`` / ``"mock"`` / ``"error"``.
            service:           Originating service name.
            brand:             LoRA brand key (if applicable).
            conditioning_mode: ControlNet mode (if applicable).
            resolution:        Resolution string e.g. ``"1024×1024"``.
            seed:              Random seed used.
            steps:             Number of inference steps.
            guidance_scale:    CFG scale used.
            lora_scale:        LoRA adapter scale used.
            latency_ms:        End-to-end generation latency in milliseconds.
            tags:              List of string tags.
            rating:            User rating 1–5 (0 = unrated).
            notes:             Free-text user notes.
            extra_meta:        Reserved for future use (ignored currently).

        Returns:
            The newly created ``HistoryEntry`` (already persisted to disk).

        Raises:
            ValueError: If ``prompt`` is blank.
        """
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        # Auto-generate tags from prompt keywords
        auto_tags = self._extract_tags(prompt, style, brand, mode)
        merged_tags = list(dict.fromkeys((tags or []) + auto_tags))  # dedup, preserve order

        entry = HistoryEntry(
            prompt           = prompt.strip(),
            negative_prompt  = negative_prompt.strip(),
            image_path       = str(image_path),
            model            = model,
            style            = style,
            mode             = mode,
            service          = service,
            brand            = brand,
            conditioning_mode= conditioning_mode,
            resolution       = resolution,
            seed             = seed,
            steps            = steps,
            guidance_scale   = guidance_scale,
            lora_scale       = lora_scale,
            latency_ms       = round(latency_ms, 1),
            tags             = merged_tags,
            rating           = max(0, min(5, int(rating))),
            notes            = notes.strip(),
        )

        with self._lock:
            # Enforce capacity cap
            if len(self._records) >= self._max:
                excess = len(self._records) - self._max + 1
                self._evict_oldest(excess)
                self._dirty = True

            self._records[entry.id] = entry
            self._append_to_file(entry)

        logger.debug(
            "HistoryManager.record | id=%s | service=%s | model=%s | latency=%.0fms",
            entry.id, service, model, latency_ms,
        )
        return entry

    # ── View / listing ─────────────────────────────────────────────────────────

    def list_entries(
        self,
        *,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
        sort: str = "newest",
        filters: Optional[Dict[str, Any]] = None,
        include_deleted: bool = False,
    ) -> Tuple[List[HistoryEntry], int]:
        """Return a paginated, filtered, sorted list of history entries.

        Args:
            page:            1-indexed page number.
            page_size:       Number of entries per page [1, 200].
            sort:            Sort order key.  One of: ``newest`` (default),
                             ``oldest``, ``prompt``, ``model``, ``style``,
                             ``rating``, ``latency``.
            filters:         Optional filter dict (see ``HistoryEntry.matches_filter``).
            include_deleted: If ``True``, include soft-deleted records.

        Returns:
            Tuple of ``(page_entries, total_matching_count)``.
        """
        page_size = max(1, min(200, page_size))
        sort_fn   = _SORT_KEYS.get(sort, _SORT_KEYS["newest"])
        filt      = filters or {}

        with self._lock:
            entries = [
                e for e in self._records.values()
                if (include_deleted or not e.deleted) and e.matches_filter(filt)
            ]

        entries.sort(key=sort_fn)
        total  = len(entries)
        start  = (page - 1) * page_size
        return entries[start: start + page_size], total

    def get_entry(self, entry_id: str) -> Optional[HistoryEntry]:
        """Return a single entry by its ID, or ``None`` if not found.

        Args:
            entry_id: UUID hex string.

        Returns:
            ``HistoryEntry`` or ``None``.
        """
        with self._lock:
            return self._records.get(entry_id)

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
        sort: str = "newest",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[HistoryEntry], int]:
        """Full-text search across all text fields.

        Searches ``prompt``, ``negative_prompt``, ``model``, ``style``,
        ``brand``, ``conditioning_mode``, ``notes``, and ``tags``.

        Args:
            query:     Search string (case-insensitive, substring match).
                       Empty string returns all entries (filtered by *filters*).
            page:      1-indexed page number.
            page_size: Entries per page.
            sort:      Sort order key.
            filters:   Additional field-level filters applied after text search.

        Returns:
            Tuple of ``(matching_entries_for_page, total_matching_count)``.
        """
        query = query.strip()
        filt  = filters or {}

        with self._lock:
            candidates = [
                e for e in self._records.values()
                if not e.deleted
                and (not query or e.matches_text(query))
                and e.matches_filter(filt)
            ]

        sort_fn = _SORT_KEYS.get(sort, _SORT_KEYS["newest"])
        candidates.sort(key=sort_fn)
        total = len(candidates)
        start = (page - 1) * max(1, page_size)
        return candidates[start: start + page_size], total

    def search_by_field(
        self,
        field: str,
        value: str,
        *,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> Tuple[List[HistoryEntry], int]:
        """Exact field-value search.

        A convenience shorthand for ``search(filters={field: value})``.

        Args:
            field:     Field name (see ``HistoryEntry.matches_filter`` for valid keys).
            value:     Value to match.
            page:      1-indexed page number.
            page_size: Entries per page.

        Returns:
            Tuple of ``(entries, total)``.
        """
        return self.search("", page=page, page_size=page_size, filters={field: value})

    # ── Update ─────────────────────────────────────────────────────────────────

    def update_entry(
        self,
        entry_id: str,
        *,
        rating: Optional[int] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[HistoryEntry]:
        """Update mutable fields of an existing entry.

        Only ``rating``, ``notes``, and ``tags`` are user-editable after creation.
        Triggers a full file rewrite to persist the change.

        Args:
            entry_id: UUID hex of the record to update.
            rating:   New rating [0–5] (``None`` = leave unchanged).
            notes:    New notes string (``None`` = leave unchanged).
            tags:     New tags list (``None`` = leave unchanged).

        Returns:
            The updated ``HistoryEntry``, or ``None`` if not found.
        """
        with self._lock:
            entry = self._records.get(entry_id)
            if entry is None:
                logger.warning("HistoryManager.update_entry: id %s not found", entry_id)
                return None

            if rating is not None:
                entry.rating = max(0, min(5, int(rating)))
            if notes is not None:
                entry.notes = str(notes).strip()
            if tags is not None:
                entry.tags = list(tags)

            self._rewrite_file()

        logger.debug("HistoryManager.update_entry | id=%s", entry_id)
        return entry

    # ── Delete ─────────────────────────────────────────────────────────────────

    def delete_entry(
        self,
        entry_id: str,
        *,
        delete_file: bool = False,
    ) -> bool:
        """Hard-delete a single record by ID.

        Args:
            entry_id:    UUID hex of the record to remove.
            delete_file: If ``True``, also delete the image file from disk.

        Returns:
            ``True`` if the record was found and deleted; ``False`` otherwise.
        """
        with self._lock:
            entry = self._records.pop(entry_id, None)
            if entry is None:
                logger.warning("HistoryManager.delete_entry: id %s not found", entry_id)
                return False

            if delete_file and entry.image_path:
                _safe_delete_file(entry.image_path)
                _safe_delete_file(entry.image_path.replace(".png", ".json"))

            self._rewrite_file()

        logger.info("HistoryManager.delete_entry | id=%s | delete_file=%s", entry_id, delete_file)
        return True

    def delete_many(
        self,
        entry_ids: List[str],
        *,
        delete_files: bool = False,
    ) -> int:
        """Hard-delete multiple records by ID list.

        Args:
            entry_ids:    List of UUID hex strings to remove.
            delete_files: If ``True``, also delete associated image files.

        Returns:
            Number of records actually deleted.
        """
        deleted = 0
        with self._lock:
            for eid in entry_ids:
                entry = self._records.pop(eid, None)
                if entry:
                    if delete_files and entry.image_path:
                        _safe_delete_file(entry.image_path)
                        _safe_delete_file(entry.image_path.replace(".png", ".json"))
                    deleted += 1
            if deleted:
                self._rewrite_file()

        logger.info("HistoryManager.delete_many | n=%d", deleted)
        return deleted

    def delete_by_filter(
        self,
        filters: Dict[str, Any],
        *,
        delete_files: bool = False,
    ) -> int:
        """Delete all records matching a filter dict.

        Args:
            filters:      Filter predicate dict (same schema as ``list_entries``).
            delete_files: If ``True``, also delete associated image files.

        Returns:
            Number of records deleted.
        """
        with self._lock:
            to_delete = [
                e.id for e in self._records.values()
                if e.matches_filter(filters)
            ]
        return self.delete_many(to_delete, delete_files=delete_files)

    def clear_all(self, *, delete_files: bool = False) -> int:
        """Delete every record in the history store.

        Args:
            delete_files: If ``True``, also delete all image files referenced.

        Returns:
            Number of records deleted.
        """
        with self._lock:
            count = len(self._records)
            if delete_files:
                for entry in self._records.values():
                    if entry.image_path:
                        _safe_delete_file(entry.image_path)
                        _safe_delete_file(entry.image_path.replace(".png", ".json"))
            self._records.clear()
            self._rewrite_file()

        logger.info("HistoryManager.clear_all | deleted=%d", count)
        return count

    # ── Prune ──────────────────────────────────────────────────────────────────

    def prune_old(
        self,
        ttl_days: int = DEFAULT_TTL_DAYS,
        *,
        delete_files: bool = False,
    ) -> int:
        """Remove records older than *ttl_days* days.

        Args:
            ttl_days:     Age threshold in days.
            delete_files: If ``True``, also delete image files.

        Returns:
            Number of records pruned.
        """
        cutoff = time.time() - ttl_days * 86400
        return self.delete_by_filter({"date_to": cutoff}, delete_files=delete_files)

    # ── Statistics ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregated usage statistics over all non-deleted records.

        Returns:
            Dict with keys:
            ``total``, ``by_model``, ``by_style``, ``by_mode``, ``by_service``,
            ``by_brand``, ``by_resolution``, ``avg_latency_ms``,
            ``top_tags``, ``rated_count``, ``avg_rating``,
            ``oldest_ts``, ``newest_ts``, ``images_on_disk``.
        """
        with self._lock:
            entries = list(self._records.values())

        if not entries:
            return {
                "total": 0, "by_model": {}, "by_style": {}, "by_mode": {},
                "by_service": {}, "by_brand": {}, "by_resolution": {},
                "avg_latency_ms": 0.0, "top_tags": [], "rated_count": 0,
                "avg_rating": 0.0, "oldest_ts": None, "newest_ts": None,
                "images_on_disk": 0,
            }

        by_model:      Dict[str, int] = {}
        by_style:      Dict[str, int] = {}
        by_mode:       Dict[str, int] = {}
        by_service:    Dict[str, int] = {}
        by_brand:      Dict[str, int] = {}
        by_resolution: Dict[str, int] = {}
        tag_counts:    Dict[str, int] = {}
        latencies:     List[float]    = []
        ratings:       List[int]      = []
        images_ok      = 0

        for e in entries:
            _inc(by_model,      e.model or "unknown")
            _inc(by_style,      e.style or "none")
            _inc(by_mode,       e.mode  or "unknown")
            _inc(by_service,    e.service or "unknown")
            if e.brand:
                _inc(by_brand, e.brand)
            if e.resolution:
                _inc(by_resolution, e.resolution)
            for tag in e.tags:
                _inc(tag_counts, tag)
            if e.latency_ms > 0:
                latencies.append(e.latency_ms)
            if e.rating > 0:
                ratings.append(e.rating)
            if e.image_exists:
                images_ok += 1

        top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:20]
        timestamps = [e.created_ts for e in entries]

        return {
            "total":           len(entries),
            "by_model":        dict(sorted(by_model.items(),      key=lambda x: -x[1])),
            "by_style":        dict(sorted(by_style.items(),      key=lambda x: -x[1])),
            "by_mode":         dict(sorted(by_mode.items(),       key=lambda x: -x[1])),
            "by_service":      dict(sorted(by_service.items(),    key=lambda x: -x[1])),
            "by_brand":        dict(sorted(by_brand.items(),      key=lambda x: -x[1])),
            "by_resolution":   dict(sorted(by_resolution.items(), key=lambda x: -x[1])),
            "avg_latency_ms":  round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "top_tags":        [{"tag": t, "count": c} for t, c in top_tags],
            "rated_count":     len(ratings),
            "avg_rating":      round(sum(ratings) / len(ratings), 2) if ratings else 0.0,
            "oldest_ts":       min(timestamps),
            "newest_ts":       max(timestamps),
            "images_on_disk":  images_ok,
        }

    # ── Export ─────────────────────────────────────────────────────────────────

    def export_json(
        self,
        *,
        query: str = "",
        filters: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None,
        indent: int = 2,
    ) -> Path:
        """Export matching history records to a JSON file.

        Args:
            query:       Optional full-text filter (empty = all records).
            filters:     Optional field-level filters.
            output_path: Override the default export path.
            indent:      JSON indentation level.

        Returns:
            Path to the written JSON file.
        """
        entries, _ = self.search(query, page=1, page_size=99999, filters=filters)
        out = output_path or (
            _EXPORTS_DIR / f"history_export_{_ts_slug()}.json"
        )
        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total":       len(entries),
            "query":       query,
            "filters":     filters or {},
            "records":     [e.to_dict() for e in entries],
        }
        out.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")
        logger.info("HistoryManager.export_json | n=%d | path=%s", len(entries), out)
        return out

    def export_csv(
        self,
        *,
        query: str = "",
        filters: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Export matching history records to a CSV file.

        Args:
            query:       Optional full-text filter.
            filters:     Optional field-level filters.
            output_path: Override the default export path.

        Returns:
            Path to the written CSV file.
        """
        entries, _ = self.search(query, page=1, page_size=99999, filters=filters)
        out = output_path or (
            _EXPORTS_DIR / f"history_export_{_ts_slug()}.csv"
        )
        _CSV_FIELDS = [
            "id", "created_at", "prompt", "model", "style", "mode", "service",
            "brand", "conditioning_mode", "resolution", "seed", "steps",
            "guidance_scale", "lora_scale", "latency_ms", "rating", "tags",
            "notes", "image_path",
        ]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for e in entries:
                row = e.to_dict()
                row["tags"] = "|".join(row.get("tags", []))
                writer.writerow(row)

        logger.info("HistoryManager.export_csv | n=%d | path=%s", len(entries), out)
        return out

    def export_markdown(
        self,
        *,
        query: str = "",
        filters: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None,
        include_stats: bool = True,
    ) -> Path:
        """Export matching history records as a Markdown report.

        Args:
            query:         Optional full-text filter.
            filters:       Optional field-level filters.
            output_path:   Override the default export path.
            include_stats: If ``True``, prepend a statistics summary table.

        Returns:
            Path to the written Markdown file.
        """
        entries, total = self.search(query, page=1, page_size=99999, filters=filters)
        out = output_path or (
            _EXPORTS_DIR / f"history_report_{_ts_slug()}.md"
        )
        lines: List[str] = [
            "# AI Fashion Studio — Generation History Report",
            "",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Total records exported:** {total}",
        ]
        if query:
            lines.append(f"**Search query:** `{query}`")
        if filters:
            lines.append(f"**Filters:** `{filters}`")
        lines.append("")

        if include_stats and entries:
            stats = self.get_stats()
            lines += [
                "## Summary Statistics",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Total generations | {stats['total']} |",
                f"| Images on disk | {stats['images_on_disk']} |",
                f"| Average latency | {stats['avg_latency_ms']} ms |",
                f"| Rated entries | {stats['rated_count']} |",
                f"| Average rating | {stats['avg_rating']:.1f} / 5 |",
                "",
                "### By Model",
                "",
            ]
            for model, count in stats["by_model"].items():
                lines.append(f"- **{model}**: {count}")
            lines += ["", "### By Style", ""]
            for style, count in list(stats["by_style"].items())[:10]:
                lines.append(f"- **{style}**: {count}")
            lines += ["", "### Top Tags", ""]
            for item in stats["top_tags"][:10]:
                lines.append(f"- `{item['tag']}` × {item['count']}")
            lines += ["", "---", ""]

        lines += ["## Generation Records", ""]
        for i, e in enumerate(entries, 1):
            stars = ("⭐" * e.rating) if e.rating else "_unrated_"
            image_link = f"[View Image]({e.image_path})" if e.image_path else "_no image_"
            lines += [
                f"### {i}. {e.short_prompt}",
                "",
                f"| Field | Value |",
                f"|-------|-------|",
                f"| **ID** | `{e.id}` |",
                f"| **Created** | {e.created_at[:19].replace('T', ' ')} UTC |",
                f"| **Model** | {e.model} |",
                f"| **Style** | {e.style or '_none_'} |",
                f"| **Mode** | `{e.mode}` |",
                f"| **Service** | {e.service} |",
                f"| **Brand** | {e.brand or '_n/a_'} |",
                f"| **Resolution** | {e.resolution or '_unknown_'} |",
                f"| **Seed** | {e.seed if e.seed is not None else '_random_'} |",
                f"| **Steps** | {e.steps} |",
                f"| **CFG** | {e.guidance_scale} |",
                f"| **Latency** | {e.latency_ms:.0f} ms |",
                f"| **Rating** | {stars} |",
                f"| **Image** | {image_link} |",
                "",
                f"**Prompt:** {e.prompt}",
            ]
            if e.negative_prompt:
                lines.append(f"**Negative:** {e.negative_prompt}")
            if e.tags:
                lines.append(f"**Tags:** {', '.join(f'`{t}`' for t in e.tags)}")
            if e.notes:
                lines.append(f"**Notes:** {e.notes}")
            lines.append("")

        out.write_text("\n".join(lines), encoding="utf-8")
        logger.info("HistoryManager.export_markdown | n=%d | path=%s", len(entries), out)
        return out

    # ── Rebuild / maintenance ──────────────────────────────────────────────────

    def rebuild_index(self) -> int:
        """Reload all records from disk, discarding the current in-memory state.

        Useful after external modifications to the JSONL file.

        Returns:
            Number of records loaded.
        """
        with self._lock:
            self._load()
        logger.info("HistoryManager.rebuild_index | loaded=%d", len(self._records))
        return len(self._records)

    def compact(self) -> Tuple[int, int]:
        """Compact the JSONL file by rewriting only live records.

        Removes lines corresponding to deleted records that may have
        accumulated via soft-delete or crash recovery.

        Returns:
            Tuple of ``(records_kept, file_size_bytes)``.
        """
        with self._lock:
            self._rewrite_file()
            size = self._file.stat().st_size if self._file.exists() else 0
        logger.info(
            "HistoryManager.compact | kept=%d | size=%d bytes",
            len(self._records), size,
        )
        return len(self._records), size

    # ── Convenience iteration ──────────────────────────────────────────────────

    def __iter__(self) -> Iterator[HistoryEntry]:
        """Iterate over all non-deleted records (newest first)."""
        with self._lock:
            entries = sorted(self._records.values(), key=lambda e: -e.created_ts)
        return iter(entries)

    def __len__(self) -> int:
        """Return the number of non-deleted records in memory."""
        with self._lock:
            return len(self._records)

    def __contains__(self, entry_id: str) -> bool:
        """Return ``True`` if *entry_id* exists in the store."""
        with self._lock:
            return entry_id in self._records

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_tags(
        prompt: str,
        style: str,
        brand: str,
        mode: str,
    ) -> List[str]:
        """Extract auto-tags from the prompt and metadata.

        Identifies fabric, garment, colour, and occasion keywords from the
        prompt text, and adds ``style:``, ``brand:``, and ``mode:`` prefix tags.
        """
        tags: List[str] = []

        _FABRIC_KEYWORDS  = {"silk","linen","cotton","denim","wool","cashmere","leather","velvet","satin","chiffon","polyester","nylon"}
        _GARMENT_KEYWORDS = {"dress","shirt","trousers","pants","jacket","coat","blazer","skirt","hoodie","sweater","top","blouse","suit","jeans","shorts","gown"}
        _COLOR_KEYWORDS   = {"black","white","navy","beige","red","blue","green","pink","grey","gray","cream","ivory","brown","gold","silver","coral","amber","teal","purple"}
        _OCCASION_KEYWORDS= {"casual","formal","sport","beach","office","evening","wedding","party","streetwear","runway","editorial"}

        words = set(re.sub(r"[^a-z0-9 ]", " ", prompt.lower()).split())

        for kw_set, prefix in [
            (_FABRIC_KEYWORDS,   "fabric"),
            (_GARMENT_KEYWORDS,  "garment"),
            (_COLOR_KEYWORDS,    "color"),
            (_OCCASION_KEYWORDS, "occasion"),
        ]:
            for kw in kw_set & words:
                tags.append(f"{prefix}:{kw}")

        if style:
            tags.append(f"style:{style.lower().replace(' ', '_')}")
        if brand:
            tags.append(f"brand:{brand.lower()}")
        if mode:
            tags.append(f"mode:{mode}")

        return tags


# ── Module-level singleton ─────────────────────────────────────────────────────

_default_manager: Optional[HistoryManager] = None
_manager_lock = threading.Lock()


def get_history_manager() -> HistoryManager:
    """Return the module-level singleton ``HistoryManager`` instance.

    Creates the instance on first call (lazy init, thread-safe).

    Returns:
        The shared ``HistoryManager``.
    """
    global _default_manager
    if _default_manager is None:
        with _manager_lock:
            if _default_manager is None:
                _default_manager = HistoryManager()
    return _default_manager


# ── Private utilities ──────────────────────────────────────────────────────────

def _inc(d: Dict[str, int], key: str) -> None:
    """Increment counter dict entry."""
    d[key] = d.get(key, 0) + 1


def _ts_slug() -> str:
    """Return a compact UTC timestamp string suitable for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_delete_file(path: str) -> None:
    """Delete a file silently, ignoring errors."""
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
    except Exception as exc:
        logger.debug("HistoryManager: could not delete file %s — %s", path, exc)

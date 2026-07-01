"""
Week 6 — State Management System.

Thread-safe, persistent session state manager for the AI Fashion Creative Studio.
Manages per-session state slices and cross-tab interaction continuity.

Responsibilities
----------------
- **Session lifecycle**  : create, retrieve, and delete user sessions.
- **Active model**       : track which SDXL model is currently selected.
- **Selected LoRA**      : track which LoRA adapter + weight is applied.
- **Active page**        : remember which studio tab a session is on.
- **Temporary outputs**  : register and clean up ephemeral files (previews,
                           ZIP exports, temp renders) when a session ends.

Architecture
------------
All state is held in an in-memory dict of ``SessionState`` objects, protected
by a ``threading.Lock``.  An optional JSON snapshot can be persisted to disk
for inspection / debugging via ``dump_snapshot()``.

Usage
-----
::

    from week6.services.state_manager import StateManager

    mgr = StateManager()

    # Create or retrieve a session
    res = mgr.get_or_create_session("user-abc")
    session = res.data          # SessionState dataclass

    # Update model selection
    mgr.set_active_model("user-abc", "SDXL v1.0 Turbo")

    # Select a LoRA adapter
    mgr.set_selected_lora("user-abc", "nike_streetwear_v2", weight=0.75)

    # Track the current page
    mgr.set_active_page("user-abc", "🎨 Text-to-Fashion")

    # Register a temp file that should be cleaned up on session end
    mgr.add_temporary_output("user-abc", "/tmp/preview_42.png")

    # Clean up temp files without ending session
    mgr.clear_temporary_outputs("user-abc")

    # Delete the session (also wipes temp files)
    mgr.delete_session("user-abc")
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from week6.gradio_app.logger import get_logger
from week6.services.base import BaseService, ServiceResult, ServiceStatus, ValidationError

logger = get_logger(__name__)

# ── Output / snapshot directory ───────────────────────────────────────────────
_SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "state"
_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Allowed values ─────────────────────────────────────────────────────────────
_VALID_PAGES = [
    "🏠 Home Dashboard",
    "🎨 Text-to-Fashion",
    "✏️ Sketch2Design",
    "🏷️ Style Switching",
    "🎛️ Style Mixer",
    "💬 Fashion Assistant",
    "👗 Recommendations",
    "🖼️ Gallery",
    "⚙️ Settings",
]

_VALID_MODELS = [
    "SDXL v1.0 (Mock)",
    "SDXL v1.0 Turbo (Mock)",
    "SDXL Lightning (Mock)",
    "SDXL v1.0",
    "SDXL v1.0 Turbo",
    "SDXL Lightning",
]

_VALID_LORAS = [
    "",                      # No LoRA selected
    "nike_streetwear_v2",
    "gucci_luxury_v1",
    "zara_minimal_v1",
    "hm_casual_v1",
    "prada_couture_v1",
    "custom_style_blend",
]


# ══════════════════════════════════════════════════════════════════════════════
# Session State Dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SessionState:
    """Per-user session state snapshot.

    Attributes:
        session_id:       Unique identifier for the session.
        user_id:          Optional human-readable user identifier.
        active_model:     Currently selected generation model name.
        selected_lora:    Active LoRA adapter name (empty string = none).
        lora_weight:      LoRA blending weight in ``[0.0, 1.0]``.
        active_page:      Current studio page/tab name.
        generation_count: Number of images generated in this session.
        chat_turns:       Number of assistant chat turns in this session.
        temporary_outputs: Paths of temporary files pending cleanup.
        last_prompt:      Most recent generation prompt text.
        last_image_path:  Path of the most recently generated image.
        preferences:      Arbitrary key-value user preference overrides.
        created_at:       Unix timestamp of session creation.
        updated_at:       Unix timestamp of last mutation.
    """

    session_id: str
    user_id: str = "demo_user"

    # ── Active selections ────────────────────────────────────────────────────
    active_model: str = "SDXL v1.0 (Mock)"
    selected_lora: str = ""
    lora_weight: float = 0.75
    active_page: str = "🏠 Home Dashboard"

    # ── Session counters ─────────────────────────────────────────────────────
    generation_count: int = 0
    chat_turns: int = 0

    # ── Temporary output tracking ────────────────────────────────────────────
    temporary_outputs: List[str] = field(default_factory=list)

    # ── Last generation context ──────────────────────────────────────────────
    last_prompt: str = ""
    last_image_path: Optional[str] = None

    # ── Arbitrary user preferences dict ─────────────────────────────────────
    preferences: Dict[str, Any] = field(default_factory=dict)

    # ── Timestamps ───────────────────────────────────────────────────────────
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def touch(self) -> None:
        """Refresh the ``updated_at`` timestamp."""
        self.updated_at = time.time()

    @property
    def age_seconds(self) -> float:
        """Seconds elapsed since the session was created."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Seconds elapsed since the session was last mutated."""
        return time.time() - self.updated_at

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the session to a JSON-safe dictionary."""
        return {
            "session_id":       self.session_id,
            "user_id":          self.user_id,
            "active_model":     self.active_model,
            "selected_lora":    self.selected_lora,
            "lora_weight":      round(self.lora_weight, 4),
            "active_page":      self.active_page,
            "generation_count": self.generation_count,
            "chat_turns":       self.chat_turns,
            "temporary_outputs": list(self.temporary_outputs),
            "last_prompt":      self.last_prompt,
            "last_image_path":  self.last_image_path,
            "preferences":      dict(self.preferences),
            "created_at":       self.created_at,
            "updated_at":       self.updated_at,
            "age_seconds":      round(self.age_seconds, 1),
            "idle_seconds":     round(self.idle_seconds, 1),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Restore a ``SessionState`` from a serialized dict."""
        obj = cls(session_id=data["session_id"])
        obj.user_id          = data.get("user_id", "demo_user")
        obj.active_model     = data.get("active_model", "SDXL v1.0 (Mock)")
        obj.selected_lora    = data.get("selected_lora", "")
        obj.lora_weight      = float(data.get("lora_weight", 0.75))
        obj.active_page      = data.get("active_page", "🏠 Home Dashboard")
        obj.generation_count = int(data.get("generation_count", 0))
        obj.chat_turns       = int(data.get("chat_turns", 0))
        obj.temporary_outputs = list(data.get("temporary_outputs", []))
        obj.last_prompt      = data.get("last_prompt", "")
        obj.last_image_path  = data.get("last_image_path")
        obj.preferences      = dict(data.get("preferences", {}))
        obj.created_at       = float(data.get("created_at", time.time()))
        obj.updated_at       = float(data.get("updated_at", time.time()))
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# StateManager Service
# ══════════════════════════════════════════════════════════════════════════════

class StateManager(BaseService):
    """Thread-safe session state manager for the AI Fashion Creative Studio.

    Manages per-user session state slices including active model selection,
    LoRA adapter, active studio page, and temporary output file tracking.

    Args:
        mock_mode:       If ``True`` validation of model/LoRA names is relaxed
                         to accept any string (useful in testing).
        session_ttl:     Seconds after which an idle session is eligible for
                         eviction.  Defaults to 3 600 s (1 hour).
        max_sessions:    Maximum number of concurrent in-memory sessions before
                         LRU eviction kicks in.  Defaults to ``200``.
        snapshot_path:   Optional path for a JSON snapshot dump.  Defaults to
                         ``outputs/state/state_snapshot.json``.
    """

    _SERVICE_NAME: str = "StateManager"

    def __init__(
        self,
        *,
        mock_mode: bool = True,
        session_ttl: float = 3600.0,
        max_sessions: int = 200,
        snapshot_path: Optional[Path] = None,
    ) -> None:
        super().__init__(mock_mode=mock_mode)
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._session_ttl  = session_ttl
        self._max_sessions = max_sessions
        self._snapshot_path = snapshot_path or (_SNAPSHOT_DIR / "state_snapshot.json")
        logger.info(
            "StateManager initialized — TTL=%.0fs  max_sessions=%d",
            self._session_ttl,
            self._max_sessions,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Session Lifecycle
    # ══════════════════════════════════════════════════════════════════════════

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        *,
        user_id: str = "demo_user",
    ) -> ServiceResult[SessionState]:
        """Retrieve an existing session or create a fresh one.

        Args:
            session_id: Caller-supplied ID.  If ``None`` a UUID4 is generated.
            user_id:    Human-readable user label stored on the session.

        Returns:
            ``ServiceResult`` whose ``.data`` is the ``SessionState``.
        """
        with self._timer() as t:
            sid = session_id or str(uuid.uuid4())
            with self._lock:
                if sid in self._sessions:
                    session = self._sessions[sid]
                    session.touch()
                    created = False
                else:
                    # Evict stale sessions if at capacity
                    self._evict_if_needed()
                    session = SessionState(session_id=sid, user_id=user_id)
                    self._sessions[sid] = session
                    created = True
                    self._call_count += 1

        action = "created" if created else "retrieved"
        logger.info("Session %s %s (user=%s)", sid, action, user_id)
        return ServiceResult.ok(
            data=session,
            meta={
                "session_id": sid,
                "created":    created,
                "user_id":    user_id,
            },
            latency_ms=t.elapsed_ms,
        )

    def get_session(
        self, session_id: str
    ) -> ServiceResult[SessionState]:
        """Retrieve an existing session without creating one.

        Returns:
            ``ServiceResult`` with ``.data = SessionState`` on hit, or a
            validation-fail result when session not found.
        """
        with self._timer() as t:
            with self._lock:
                session = self._sessions.get(session_id)

        if session is None:
            return ServiceResult.validation_fail(
                "session_id", f"No session found for id={session_id!r}"
            )
        return ServiceResult.ok(
            data=session,
            meta={"session_id": session_id},
            latency_ms=t.elapsed_ms,
        )

    def delete_session(
        self, session_id: str
    ) -> ServiceResult[Dict[str, Any]]:
        """Delete a session and clean up all its temporary output files.

        Args:
            session_id: Session to remove.

        Returns:
            ``ServiceResult`` with summary of files cleaned up.
        """
        with self._timer() as t:
            with self._lock:
                session = self._sessions.pop(session_id, None)

        if session is None:
            return ServiceResult.validation_fail(
                "session_id", f"No session found for id={session_id!r}"
            )

        # Cleanup temp files outside the lock (I/O can be slow)
        cleaned, failed = self._delete_temp_files(session.temporary_outputs)

        logger.info(
            "Session %s deleted — %d temp files cleaned, %d failed",
            session_id, cleaned, failed,
        )
        return ServiceResult.ok(
            data={"session_id": session_id, "files_cleaned": cleaned, "files_failed": failed},
            latency_ms=t.elapsed_ms,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # State Mutations
    # ══════════════════════════════════════════════════════════════════════════

    def set_active_model(
        self, session_id: str, model: str
    ) -> ServiceResult[str]:
        """Update the active generation model for a session.

        Args:
            session_id: Target session.
            model:      Model name to activate.

        Returns:
            ``ServiceResult[str]`` with the accepted model name.
        """
        with self._timer() as t:
            # Validate
            if not model or not isinstance(model, str):
                return ServiceResult.validation_fail("model", "must be a non-empty string")
            model = model.strip()
            if not self.mock_mode and model not in _VALID_MODELS:
                return ServiceResult.validation_fail(
                    "model",
                    f"must be one of {_VALID_MODELS!r}, got {model!r}",
                )

            with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    return ServiceResult.validation_fail(
                        "session_id", f"No session found for id={session_id!r}"
                    )
                old = session.active_model
                session.active_model = model
                session.touch()
                self._call_count += 1

        logger.info("Session %s model: %r → %r", session_id, old, model)
        return ServiceResult.ok(
            data=model,
            meta={"session_id": session_id, "previous_model": old},
            latency_ms=t.elapsed_ms,
        )

    def set_selected_lora(
        self,
        session_id: str,
        lora: str,
        *,
        weight: float = 0.75,
    ) -> ServiceResult[Dict[str, Any]]:
        """Update the selected LoRA adapter and blending weight.

        Args:
            session_id: Target session.
            lora:       LoRA adapter name.  Pass ``""`` to deselect.
            weight:     LoRA blending strength in ``[0.0, 1.0]``.

        Returns:
            ``ServiceResult[dict]`` with ``lora`` and ``weight`` keys.
        """
        with self._timer() as t:
            if not isinstance(lora, str):
                return ServiceResult.validation_fail("lora", "must be a string")
            lora = lora.strip()
            if not self.mock_mode and lora and lora not in _VALID_LORAS:
                return ServiceResult.validation_fail(
                    "lora",
                    f"must be one of {_VALID_LORAS!r} or empty, got {lora!r}",
                )
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                return ServiceResult.validation_fail("weight", "must be a float")
            if not (0.0 <= weight <= 1.0):
                return ServiceResult.validation_fail(
                    "weight", f"must be in [0.0, 1.0], got {weight}"
                )

            with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    return ServiceResult.validation_fail(
                        "session_id", f"No session found for id={session_id!r}"
                    )
                old_lora   = session.selected_lora
                old_weight = session.lora_weight
                session.selected_lora = lora
                session.lora_weight   = weight
                session.touch()
                self._call_count += 1

        logger.info(
            "Session %s LoRA: %r@%.2f → %r@%.2f",
            session_id, old_lora, old_weight, lora, weight,
        )
        return ServiceResult.ok(
            data={"lora": lora, "weight": weight},
            meta={
                "session_id":    session_id,
                "previous_lora": old_lora,
                "previous_weight": old_weight,
            },
            latency_ms=t.elapsed_ms,
        )

    def set_active_page(
        self, session_id: str, page: str
    ) -> ServiceResult[str]:
        """Update the currently active studio page/tab.

        Args:
            session_id: Target session.
            page:       Page name matching one of the sidebar tab labels.

        Returns:
            ``ServiceResult[str]`` with the accepted page name.
        """
        with self._timer() as t:
            if not page or not isinstance(page, str):
                return ServiceResult.validation_fail("page", "must be a non-empty string")
            page = page.strip()
            if not self.mock_mode and page not in _VALID_PAGES:
                return ServiceResult.validation_fail(
                    "page",
                    f"must be one of {_VALID_PAGES!r}, got {page!r}",
                )

            with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    return ServiceResult.validation_fail(
                        "session_id", f"No session found for id={session_id!r}"
                    )
                old = session.active_page
                session.active_page = page
                session.touch()
                self._call_count += 1

        logger.info("Session %s page: %r → %r", session_id, old, page)
        return ServiceResult.ok(
            data=page,
            meta={"session_id": session_id, "previous_page": old},
            latency_ms=t.elapsed_ms,
        )

    def update_generation_context(
        self,
        session_id: str,
        *,
        prompt: str = "",
        image_path: Optional[str] = None,
    ) -> ServiceResult[Dict[str, Any]]:
        """Record the last generation prompt and output image path.

        Called automatically by ``GenerationService`` / ``ControlNetService``
        to keep cross-tab context in sync.

        Args:
            session_id:  Target session.
            prompt:      The generation prompt text.
            image_path:  Absolute path of the generated image, or ``None``.

        Returns:
            ``ServiceResult[dict]`` with the saved context.
        """
        with self._timer() as t:
            with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    return ServiceResult.validation_fail(
                        "session_id", f"No session found for id={session_id!r}"
                    )
                session.last_prompt     = str(prompt).strip()
                session.last_image_path = image_path
                session.generation_count += 1
                session.touch()
                self._call_count += 1

        return ServiceResult.ok(
            data={"prompt": session.last_prompt, "image_path": image_path},
            meta={"session_id": session_id, "generation_count": session.generation_count},
            latency_ms=t.elapsed_ms,
        )

    def increment_chat_turns(self, session_id: str) -> ServiceResult[int]:
        """Increment the assistant chat turn counter for a session.

        Returns:
            ``ServiceResult[int]`` with the new turn count.
        """
        with self._timer() as t:
            with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    return ServiceResult.validation_fail(
                        "session_id", f"No session found for id={session_id!r}"
                    )
                session.chat_turns += 1
                session.touch()
                count = session.chat_turns
                self._call_count += 1

        return ServiceResult.ok(
            data=count,
            meta={"session_id": session_id},
            latency_ms=t.elapsed_ms,
        )

    def set_preference(
        self,
        session_id: str,
        key: str,
        value: Any,
    ) -> ServiceResult[Dict[str, Any]]:
        """Store an arbitrary user preference override on the session.

        Args:
            session_id: Target session.
            key:        Preference key (non-empty string).
            value:      JSON-serializable preference value.

        Returns:
            ``ServiceResult[dict]`` with ``{key: value}``.
        """
        with self._timer() as t:
            if not key or not isinstance(key, str):
                return ServiceResult.validation_fail("key", "must be a non-empty string")
            with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    return ServiceResult.validation_fail(
                        "session_id", f"No session found for id={session_id!r}"
                    )
                session.preferences[key] = value
                session.touch()
                self._call_count += 1

        return ServiceResult.ok(
            data={key: value},
            meta={"session_id": session_id},
            latency_ms=t.elapsed_ms,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Temporary Output Management
    # ══════════════════════════════════════════════════════════════════════════

    def add_temporary_output(
        self, session_id: str, file_path: str
    ) -> ServiceResult[str]:
        """Register a temporary file path to be cleaned up when the session ends.

        Args:
            session_id: Target session.
            file_path:  Absolute or relative path of the ephemeral file.

        Returns:
            ``ServiceResult[str]`` with the registered file path.
        """
        with self._timer() as t:
            if not file_path or not isinstance(file_path, str):
                return ServiceResult.validation_fail(
                    "file_path", "must be a non-empty string"
                )
            file_path = str(file_path).strip()

            with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    return ServiceResult.validation_fail(
                        "session_id", f"No session found for id={session_id!r}"
                    )
                if file_path not in session.temporary_outputs:
                    session.temporary_outputs.append(file_path)
                session.touch()
                total = len(session.temporary_outputs)
                self._call_count += 1

        logger.debug(
            "Session %s — registered temp output: %s (total=%d)",
            session_id, file_path, total,
        )
        return ServiceResult.ok(
            data=file_path,
            meta={"session_id": session_id, "total_temp_outputs": total},
            latency_ms=t.elapsed_ms,
        )

    def clear_temporary_outputs(
        self, session_id: str
    ) -> ServiceResult[Dict[str, int]]:
        """Delete all tracked temporary files for a session without removing the session.

        Args:
            session_id: Target session.

        Returns:
            ``ServiceResult[dict]`` with ``files_cleaned`` and ``files_failed`` counts.
        """
        with self._timer() as t:
            with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    return ServiceResult.validation_fail(
                        "session_id", f"No session found for id={session_id!r}"
                    )
                paths = list(session.temporary_outputs)
                session.temporary_outputs.clear()
                session.touch()

        cleaned, failed = self._delete_temp_files(paths)
        logger.info(
            "Session %s — cleared %d temp files (%d failed)",
            session_id, cleaned, failed,
        )
        return ServiceResult.ok(
            data={"files_cleaned": cleaned, "files_failed": failed},
            meta={"session_id": session_id},
            latency_ms=t.elapsed_ms,
        )

    def get_temporary_outputs(
        self, session_id: str
    ) -> ServiceResult[List[str]]:
        """Return the list of registered temporary output paths for a session.

        Args:
            session_id: Target session.

        Returns:
            ``ServiceResult[List[str]]`` with the temp file paths.
        """
        with self._timer() as t:
            with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    return ServiceResult.validation_fail(
                        "session_id", f"No session found for id={session_id!r}"
                    )
                paths = list(session.temporary_outputs)

        return ServiceResult.ok(
            data=paths,
            meta={"session_id": session_id, "count": len(paths)},
            latency_ms=t.elapsed_ms,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Session Listing & Statistics
    # ══════════════════════════════════════════════════════════════════════════

    def list_sessions(self) -> ServiceResult[List[Dict[str, Any]]]:
        """Return serialized snapshots of all active sessions.

        Returns:
            ``ServiceResult[List[dict]]`` sorted by creation time (newest first).
        """
        with self._timer() as t:
            with self._lock:
                snapshots = [s.to_dict() for s in self._sessions.values()]
        snapshots.sort(key=lambda d: d["created_at"], reverse=True)
        return ServiceResult.ok(
            data=snapshots,
            meta={"total_sessions": len(snapshots)},
            latency_ms=t.elapsed_ms,
        )

    def get_stats(self) -> ServiceResult[Dict[str, Any]]:
        """Return aggregate statistics across all active sessions.

        Returns:
            ``ServiceResult[dict]`` with counts and aggregations.
        """
        with self._timer() as t:
            with self._lock:
                sessions = list(self._sessions.values())

        total = len(sessions)
        page_counts: Dict[str, int] = {}
        model_counts: Dict[str, int] = {}
        lora_counts: Dict[str, int] = {}
        total_gen = sum(s.generation_count for s in sessions)
        total_chat = sum(s.chat_turns for s in sessions)
        total_temp = sum(len(s.temporary_outputs) for s in sessions)

        for s in sessions:
            page_counts[s.active_page]    = page_counts.get(s.active_page, 0) + 1
            model_counts[s.active_model]  = model_counts.get(s.active_model, 0) + 1
            lk = s.selected_lora or "(none)"
            lora_counts[lk] = lora_counts.get(lk, 0) + 1

        stats: Dict[str, Any] = {
            "total_sessions":       total,
            "total_generations":    total_gen,
            "total_chat_turns":     total_chat,
            "pending_temp_outputs": total_temp,
            "page_distribution":    page_counts,
            "model_distribution":   model_counts,
            "lora_distribution":    lora_counts,
        }
        return ServiceResult.ok(data=stats, latency_ms=t.elapsed_ms)

    # ══════════════════════════════════════════════════════════════════════════
    # Snapshot Persistence
    # ══════════════════════════════════════════════════════════════════════════

    def dump_snapshot(
        self, path: Optional[Path] = None
    ) -> ServiceResult[str]:
        """Serialize all sessions to a JSON snapshot file for inspection.

        Args:
            path: Output file path.  Defaults to ``outputs/state/state_snapshot.json``.

        Returns:
            ``ServiceResult[str]`` with the absolute path of the written file.
        """
        with self._timer() as t:
            out = path or self._snapshot_path
            with self._lock:
                data = {
                    "timestamp": time.time(),
                    "total":     len(self._sessions),
                    "sessions":  [s.to_dict() for s in self._sessions.values()],
                }
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except OSError as exc:
                self._error_count += 1
                return ServiceResult.fail(f"Snapshot write failed: {exc}", code="SNAPSHOT_ERROR")

        logger.info("State snapshot written to %s", out)
        return ServiceResult.ok(data=str(out), latency_ms=t.elapsed_ms)

    def load_snapshot(
        self, path: Optional[Path] = None
    ) -> ServiceResult[int]:
        """Restore sessions from a previously dumped JSON snapshot.

        Existing in-memory sessions with matching IDs are overwritten.

        Args:
            path: Source file path.  Defaults to ``outputs/state/state_snapshot.json``.

        Returns:
            ``ServiceResult[int]`` with the number of sessions loaded.
        """
        with self._timer() as t:
            src = path or self._snapshot_path
            if not src.exists():
                return ServiceResult.fail(
                    f"Snapshot file not found: {src}", code="SNAPSHOT_NOT_FOUND"
                )
            try:
                with open(src, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                self._error_count += 1
                return ServiceResult.fail(f"Snapshot read failed: {exc}", code="SNAPSHOT_ERROR")

            loaded = 0
            with self._lock:
                for raw in data.get("sessions", []):
                    try:
                        s = SessionState.from_dict(raw)
                        self._sessions[s.session_id] = s
                        loaded += 1
                    except Exception as exc:
                        logger.warning("Failed to restore session from snapshot: %s", exc)

        logger.info("Loaded %d sessions from snapshot %s", loaded, src)
        return ServiceResult.ok(data=loaded, latency_ms=t.elapsed_ms)

    # ══════════════════════════════════════════════════════════════════════════
    # Health Check
    # ══════════════════════════════════════════════════════════════════════════

    def health_check(self) -> ServiceResult:
        """Return a lightweight health summary.

        Returns:
            ServiceResult wrapping health details.
        """
        with self._lock:
            total = len(self._sessions)

        # Verify snapshot directory is writable
        writable = os.access(_SNAPSHOT_DIR, os.W_OK)
        status = "ok" if writable else "degraded"
        msg    = (
            "StateManager operational — snapshot directory readable/writable"
            if writable
            else f"StateManager degraded — snapshot dir not writable: {_SNAPSHOT_DIR}"
        )
        res = {
            "status":         status,
            "message":        msg,
            "mock_mode":      self.mock_mode,
            "total_sessions": total,
            "max_sessions":   self._max_sessions,
            "session_ttl_s":  self._session_ttl,
            "call_count":     self._call_count,
            "error_count":    self._error_count,
        }
        return ServiceResult(success=(status == "ok"), data=res)

    # ══════════════════════════════════════════════════════════════════════════
    # Private Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _delete_temp_files(self, paths: List[str]) -> tuple[int, int]:
        """Attempt to delete a list of file paths.  Returns (cleaned, failed)."""
        cleaned = failed = 0
        for p in paths:
            try:
                fp = Path(p)
                if fp.exists():
                    fp.unlink()
                cleaned += 1
            except OSError as exc:
                logger.warning("Failed to delete temp file %s: %s", p, exc)
                failed += 1
        return cleaned, failed

    def _evict_if_needed(self) -> None:
        """Evict the oldest idle sessions when ``max_sessions`` is exceeded.

        Must be called **inside** ``self._lock``.
        """
        if len(self._sessions) < self._max_sessions:
            return
        # Sort by last-updated ascending (oldest idle first)
        sorted_ids = sorted(
            self._sessions, key=lambda sid: self._sessions[sid].updated_at
        )
        evict_count = max(1, len(self._sessions) - self._max_sessions + 1)
        for sid in sorted_ids[:evict_count]:
            evicted = self._sessions.pop(sid)
            logger.info(
                "Evicted idle session %s (idle %.0fs)", sid, evicted.idle_seconds
            )

    def evict_stale_sessions(self) -> ServiceResult[int]:
        """Remove sessions that have been idle longer than ``session_ttl``.

        Returns:
            ``ServiceResult[int]`` with the count of evicted sessions.
        """
        with self._timer() as t:
            now = time.time()
            evicted_ids = []
            with self._lock:
                for sid, s in list(self._sessions.items()):
                    if (now - s.updated_at) > self._session_ttl:
                        evicted_ids.append(sid)
                for sid in evicted_ids:
                    stale = self._sessions.pop(sid)
                    # Queue temp file cleanup (files may not exist, ignore errors)
                    for p in stale.temporary_outputs:
                        try:
                            fp = Path(p)
                            if fp.exists():
                                fp.unlink()
                        except OSError:
                            pass

        logger.info("Evicted %d stale sessions (TTL=%.0fs)", len(evicted_ids), self._session_ttl)
        return ServiceResult.ok(
            data=len(evicted_ids),
            meta={"evicted_ids": evicted_ids},
            latency_ms=t.elapsed_ms,
        )

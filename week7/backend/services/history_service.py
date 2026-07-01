from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from week6.services.history_manager import HistoryManager


class HistoryService:
    """Business logic for persisting and querying design generation logs."""

    def __init__(self) -> None:
        self._mgr = HistoryManager()

    def record(
        self,
        prompt: str,
        image_path: str,
        model: str,
        style: str,
        mode: str,
        seed: int,
        resolution: str,
        latency_ms: float,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        entry = self._mgr.record(
            prompt=prompt,
            image_path=image_path,
            model=model,
            style=style,
            mode=mode,
            seed=seed,
            resolution=resolution,
            latency_ms=latency_ms,
            tags=tags
        )
        return entry.to_dict() if hasattr(entry, "to_dict") else vars(entry)

    def search(
        self,
        query: str = "",
        page: int = 1,
        page_size: int = 20,
        sort: str = "newest",
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Any], int]:
        return self._mgr.search(
            query=query,
            page=page,
            page_size=page_size,
            sort=sort,
            filters=filters
        )

    def get_entry(self, entry_id: str) -> Optional[Any]:
        return self._mgr.get_entry(entry_id)

    def update_entry(
        self,
        entry_id: str,
        rating: Optional[int] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Any]:
        return self._mgr.update_entry(
            entry_id=entry_id,
            rating=rating,
            notes=notes,
            tags=tags
        )

    def delete_entry(self, entry_id: str, delete_file: bool = False) -> bool:
        return self._mgr.delete_entry(entry_id=entry_id, delete_file=delete_file)

    def health_check(self) -> Dict[str, Any]:
        """Verify the health status of the history service."""
        return {
            "status": "ok",
            "name": "HistoryService"
        }

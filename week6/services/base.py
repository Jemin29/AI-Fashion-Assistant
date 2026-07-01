"""
Week 6 — Service Layer Base Contracts.

Defines the shared typed interface used by every service in ``week6/services/``:

- ``ServiceStatus``   — Enum of possible service / result states
- ``ServiceError``    — Typed exception for service-layer failures
- ``ServiceResult``   — Generic dataclass wrapping every service output
- ``BaseService``     — Abstract base class with health-check / status contract
- ``ValidationError`` — Raised when caller-supplied inputs fail validation

These types allow Gradio event callbacks to handle errors uniformly::

    result = generation_service.generate(...)
    if result.error:
        gr.Warning(result.error)
    else:
        return result.data["image"], result.meta

"""
from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


# ══════════════════════════════════════════════════════════════════════════════
# Status Enum
# ══════════════════════════════════════════════════════════════════════════════

class ServiceStatus(str, Enum):
    """Overall status of a service call or a service instance."""

    OK           = "ok"           # Completed successfully
    MOCK         = "mock"         # Completed in mock/fallback mode
    DEGRADED     = "degraded"     # Real backend partially available
    ERROR        = "error"        # Unrecoverable failure
    VALIDATION   = "validation"   # Caller input failed validation
    UNAVAILABLE  = "unavailable"  # Backend service not reachable


# ══════════════════════════════════════════════════════════════════════════════
# Exceptions
# ══════════════════════════════════════════════════════════════════════════════

class ServiceError(Exception):
    """Raised when a service method encounters an unrecoverable error.

    Attributes:
        message:    Human-readable error description.
        code:       Machine-readable error code (e.g. ``"PROMPT_EMPTY"``).
        service:    Name of the originating service class.
        context:    Optional dict of additional diagnostic context.
        traceback_: Captured traceback string (set automatically on ``from_exc``).
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "SERVICE_ERROR",
        service: str = "unknown",
        context: Optional[Dict[str, Any]] = None,
        traceback_: str = "",
    ) -> None:
        super().__init__(message)
        self.message   = message
        self.code      = code
        self.service   = service
        self.context   = context or {}
        self.traceback_= traceback_

    @classmethod
    def from_exc(
        cls,
        exc: Exception,
        *,
        code: str = "INTERNAL_ERROR",
        service: str = "unknown",
        context: Optional[Dict[str, Any]] = None,
    ) -> "ServiceError":
        """Wrap an arbitrary exception in a ``ServiceError``.

        Args:
            exc:     The original exception.
            code:    Machine-readable error code.
            service: Name of the originating service.
            context: Optional diagnostic context dict.

        Returns:
            A new ``ServiceError`` with traceback captured.
        """
        return cls(
            message=str(exc),
            code=code,
            service=service,
            context=context or {},
            traceback_=traceback.format_exc(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict (for logging / response meta)."""
        return {
            "error":   self.message,
            "code":    self.code,
            "service": self.service,
            "context": self.context,
        }

    def __repr__(self) -> str:
        return f"ServiceError(code={self.code!r}, service={self.service!r}, msg={self.message!r})"


class ValidationError(ServiceError):
    """Raised when caller-supplied inputs fail pre-condition validation.

    Args:
        field:   The name of the invalid field (e.g. ``"prompt"``).
        reason:  Human-readable description of why validation failed.
        service: Name of the originating service class.
    """

    def __init__(self, field: str, reason: str, *, service: str = "unknown") -> None:
        super().__init__(
            message=f"Validation failed for '{field}': {reason}",
            code="VALIDATION_ERROR",
            service=service,
            context={"field": field, "reason": reason},
        )
        self.field  = field
        self.reason = reason


# ══════════════════════════════════════════════════════════════════════════════
# Result Container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(init=False)
class ServiceResult(dict, Generic[T]):
    """Generic result container for all service method calls.

    Every service method that performs computation returns a ``ServiceResult``
    so that UI callbacks can handle success / failure uniformly.

    Attributes:
        success:     Boolean indicating whether the call succeeded.
        data:        Primary output payload (image, list, string, …).
        message:     Human-readable output message or description.
        error:       Human-readable error message (``None`` on success).
        metadata:    Dict of metadata (mode, model, latency_ms, timestamp, …).
    """

    success:     bool
    data:        Optional[T]
    message:     str
    error:       Optional[str]
    metadata:    Dict[str, Any]
    status:      ServiceStatus
    error_code:  Optional[str]
    warnings:    List[str]
    latency_ms:  float

    def __init__(
        self,
        success: bool = True,
        data: Optional[T] = None,
        message: str = "",
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: ServiceStatus = ServiceStatus.OK,
        error_code: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        latency_ms: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.status = status
        if status in (ServiceStatus.ERROR, ServiceStatus.VALIDATION, ServiceStatus.UNAVAILABLE):
            self.success = False
        else:
            self.success = success

        self.data = data
        self.error = error
        self.message = message or error or ""
        self.metadata = metadata or meta or {}
        self.error_code = error_code
        self.warnings = warnings or []
        self.latency_ms = latency_ms

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        # Keep internal dict storage synchronized with attributes
        self[name] = value

    # ── Convenience constructors ─────────────────────────────────────────────

    @classmethod
    def ok(
        cls,
        data: T,
        meta: Optional[Dict[str, Any]] = None,
        *,
        latency_ms: float = 0.0,
        warnings: Optional[List[str]] = None,
    ) -> "ServiceResult[T]":
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            meta=meta or {},
            status=ServiceStatus.OK,
            latency_ms=latency_ms,
            warnings=warnings or [],
        )

    @classmethod
    def mock(
        cls,
        data: T,
        meta: Optional[Dict[str, Any]] = None,
        *,
        latency_ms: float = 0.0,
        warnings: Optional[List[str]] = None,
    ) -> "ServiceResult[T]":
        """Create a successful mock-mode result."""
        _meta = dict(meta or {})
        _meta.setdefault("mode", "mock")
        return cls(
            success=True,
            data=data,
            meta=_meta,
            status=ServiceStatus.MOCK,
            latency_ms=latency_ms,
            warnings=warnings or [],
        )

    @classmethod
    def fail(
        cls,
        message: str,
        *,
        code: str = "ERROR",
        meta: Optional[Dict[str, Any]] = None,
        latency_ms: float = 0.0,
    ) -> "ServiceResult[None]":
        """Create a failed result (data=None, status=ERROR)."""
        return cls(
            success=False,
            data=None,
            meta=meta or {},
            status=ServiceStatus.ERROR,
            error=message,
            message=message,
            error_code=code,
            latency_ms=latency_ms,
        )

    @classmethod
    def validation_fail(
        cls,
        field: str,
        reason: str,
    ) -> "ServiceResult[None]":
        """Create a validation-failed result."""
        msg = f"Invalid '{field}': {reason}"
        return cls(
            success=False,
            data=None,
            meta={},
            status=ServiceStatus.VALIDATION,
            error=msg,
            message=msg,
            error_code="VALIDATION_ERROR",
        )

    @classmethod
    def from_service_error(cls, exc: "ServiceError") -> "ServiceResult[None]":
        """Wrap a ``ServiceError`` into a failed result."""
        return cls(
            success=False,
            data=None,
            meta=exc.to_dict(),
            status=ServiceStatus.ERROR,
            error=exc.message,
            message=exc.message,
            error_code=exc.code,
        )

    # ── State helpers ────────────────────────────────────────────────────────

    @property
    def is_ok(self) -> bool:
        """``True`` if the call succeeded (real or mock)."""
        return self.success

    @property
    def meta(self) -> Dict[str, Any]:
        """Backward compatibility mapping to metadata."""
        return self.metadata

    @meta.setter
    def meta(self, val: Dict[str, Any]) -> None:
        self.metadata = val

    @property
    def has_error(self) -> bool:
        """``True`` if the call failed."""
        return self.error is not None

    def add_warning(self, msg: str) -> None:
        """Append a non-fatal advisory warning."""
        self.warnings.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to a JSON-safe dict (excluding raw binary data)."""
        return {
            "status":     self.status.value,
            "error":      self.error,
            "error_code": self.error_code,
            "meta":       self.metadata,
            "warnings":   self.warnings,
            "latency_ms": round(self.latency_ms, 1),
        }


    def __iter__(self) -> Any:
        """Allow unpacking/iteration based on the wrapped data type."""
        if isinstance(self.data, (list, tuple)):
            return iter(self.data)
        
        # Tuple unpacking fallback: yield (image/data, meta)
        def _unpack():
            if isinstance(self.data, dict) and "image" in self.data:
                yield self.data["image"]
            else:
                yield self.data
            yield self.meta
            
        return _unpack()

    def __len__(self) -> int:
        """Return the length of the underlying data if it supports len(), else dict keys count."""
        if isinstance(self.data, (list, dict, tuple)):
            return len(self.data)
        return dict.__len__(self)

    def __contains__(self, key: Any) -> bool:
        """Check if key exists in the ServiceResult dictionary or inside data dictionary."""
        if dict.__contains__(self, key):
            return True
        if isinstance(self.data, dict) and key in self.data:
            return True
        return False

    def __getitem__(self, key: Any) -> Any:
        """Allow indexing/subscripting into the result data, metadata, or dict keys."""
        if isinstance(self.data, list) and isinstance(key, int):
            return self.data[key]
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if isinstance(self.data, dict) and key in self.data:
            return self.data[key]
        if key == 0:
            if isinstance(self.data, dict) and "image" in self.data:
                return self.data["image"]
            return self.data
        if key == 1:
            return self.meta
        raise KeyError(f"Invalid key for ServiceResult: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Allow dictionary-like .get() access, delegating to dict keys or data dict."""
        if dict.__contains__(self, key):
            return dict.get(self, key)
        if isinstance(self.data, dict) and key in self.data:
            return self.data.get(key, default)
        return getattr(self, key, default)

    def __bool__(self) -> bool:
        """Allow boolean checks. Returns True if status is OK/MOCK and data is present/not empty."""
        if not self.is_ok:
            return False
        if isinstance(self.data, (list, dict)):
            return len(self.data) > 0
        return self.data is not None



# ══════════════════════════════════════════════════════════════════════════════
# Timing Context Manager
# ══════════════════════════════════════════════════════════════════════════════

class _Timer:
    """Simple monotonic timer used internally by BaseService."""

    def __init__(self) -> None:
        self._start: float = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    @property
    def elapsed_ms(self) -> float:
        """Elapsed milliseconds since context entry."""
        return (time.monotonic() - self._start) * 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# Base Service
# ══════════════════════════════════════════════════════════════════════════════

class BaseService(ABC):
    """Abstract base class that all Week 6 services inherit from.

    Provides:
    - ``mock_mode`` flag management
    - ``_timer()`` context manager for latency measurement
    - ``health_check()`` abstract method (subclasses must implement)
    - ``status()`` property returning a human-readable dict
    - ``_validate_str()`` / ``_validate_range()`` helper validators
    """

    _SERVICE_NAME: str = "BaseService"

    def __init__(self, mock_mode: bool = True) -> None:
        self.mock_mode = mock_mode
        self._init_time = time.time()
        self._call_count: int = 0
        self._error_count: int = 0

    # ── Health / Status ──────────────────────────────────────────────────────

    @abstractmethod
    def health_check(self) -> ServiceResult:
        """Perform a lightweight health check.

        Returns:
            ServiceResult wrapping health details.
        """
        ...

    @property
    def status(self) -> Dict[str, Any]:
        """Return a snapshot of service runtime statistics."""
        return {
            "service":      self._SERVICE_NAME,
            "mock_mode":    self.mock_mode,
            "call_count":   self._call_count,
            "error_count":  self._error_count,
            "error_rate":   (
                f"{self._error_count / self._call_count:.1%}"
                if self._call_count else "n/a"
            ),
            "uptime_s":     round(time.time() - self._init_time, 1),
        }

    # ── Timing ───────────────────────────────────────────────────────────────

    def _timer(self) -> "_Timer":
        """Return a context manager that measures elapsed wall-clock time."""
        return _Timer()

    # ── Validation helpers ───────────────────────────────────────────────────

    def _validate_str(
        self,
        value: Any,
        field: str,
        *,
        min_len: int = 1,
        max_len: int = 2000,
    ) -> str:
        """Validate that *value* is a non-empty string within length bounds.

        Args:
            value:   The value to validate (will be coerced to str).
            field:   Field name for error messages.
            min_len: Minimum length (default 1).
            max_len: Maximum length (default 2000).

        Returns:
            The validated and stripped string.

        Raises:
            ValidationError: If validation fails.
        """
        if not isinstance(value, str):
            raise ValidationError(field, f"expected str, got {type(value).__name__}", service=self._SERVICE_NAME)
        v = value.strip()
        if len(v) < min_len:
            raise ValidationError(field, f"must not be empty (min {min_len} chars)", service=self._SERVICE_NAME)
        if len(v) > max_len:
            raise ValidationError(field, f"too long ({len(v)} chars, max {max_len})", service=self._SERVICE_NAME)
        return v

    def _validate_range(
        self,
        value: Any,
        field: str,
        *,
        lo: float,
        hi: float,
    ) -> float:
        """Validate that *value* is a number within [lo, hi].

        Args:
            value: The value to validate (must be int or float).
            field: Field name for error messages.
            lo:    Minimum allowed value (inclusive).
            hi:    Maximum allowed value (inclusive).

        Returns:
            The validated float.

        Raises:
            ValidationError: If validation fails.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValidationError(field, f"expected number, got {type(value).__name__}", service=self._SERVICE_NAME)
        if not (lo <= v <= hi):
            raise ValidationError(field, f"must be in [{lo}, {hi}], got {v}", service=self._SERVICE_NAME)
        return v

    def _validate_choice(
        self,
        value: Any,
        field: str,
        choices: List[str],
    ) -> str:
        """Validate that *value* is one of the allowed *choices*.

        Args:
            value:   The value to check.
            field:   Field name for error messages.
            choices: Allowed values.

        Returns:
            The validated string.

        Raises:
            ValidationError: If validation fails.
        """
        v = str(value).strip().lower()
        allowed = [c.lower() for c in choices]
        if v not in allowed:
            raise ValidationError(
                field,
                f"must be one of {choices!r}, got {value!r}",
                service=self._SERVICE_NAME,
            )
        # Return original-case match
        return choices[allowed.index(v)]

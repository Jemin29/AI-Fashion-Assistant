"""
Week 6 — Export Manager.

Handles single and batch exporting of generated fashion images, metadata,
recommendation reports, and chat conversation logs into standard formats:
PNG, JPEG, JSON, CSV, and compressed ZIP archives.

Provides a unified interface with validation, error handling, and latency tracking,
returning typed ServiceResult outputs.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from week6.services.base import BaseService, ServiceResult, ServiceStatus, ValidationError

try:
    from week6.gradio_app.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

# ── Storage configurations ─────────────────────────────────────────────────────
_EXPORTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "exports"
_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ExportManager(BaseService):
    """
    Thread-safe UI-to-backend adapter for exporting AI Fashion Studio assets.

    Supports writing images, metadata logs, recommendations, and conversational
    chat history to formats like PNG, JPEG, JSON, CSV, and packaging them
    into a ZIP file.
    """

    _SERVICE_NAME = "ExportManager"

    def __init__(self, mock_mode: bool = False) -> None:
        super().__init__(mock_mode)
        logger.info("ExportManager successfully initialized. Output directory: %s", _EXPORTS_DIR)

    # ── 1. Image Exporting ────────────────────────────────────────────────────

    def export_image(
        self,
        image: Union[Image.Image, str, Path],
        format_type: str = "PNG",
        quality: int = 95,
        filename: Optional[str] = None,
    ) -> ServiceResult[str]:
        """Convert and export a generated image to PNG or JPEG format.

        Args:
            image:       PIL Image instance, or a string/Path to an existing image file.
            format_type: Target extension/format ("PNG" or "JPEG" / "JPG").
            quality:     Compression quality for JPEG (1-100).
            filename:    Optional target filename. Defaults to a timestamped name.

        Returns:
            ServiceResult wrapping the absolute path to the exported image.
        """
        self._call_count += 1
        warnings: List[str] = []

        # Validate format
        fmt = format_type.upper().strip()
        if fmt == "JPG":
            fmt = "JPEG"
        if fmt not in ("PNG", "JPEG"):
            self._error_count += 1
            return ServiceResult.validation_fail("format_type", f"Unsupported format '{format_type}'. Must be PNG or JPEG.")

        # Validate quality
        try:
            quality = int(self._validate_range(quality, "quality", lo=1, hi=100))
        except ValidationError as e:
            warnings.append(f"Quality clamped: {e.reason}")
            quality = max(1, min(100, int(quality)))

        # Load image
        pil_img: Optional[Image.Image] = None
        with self._timer() as t:
            try:
                if isinstance(image, (str, Path)):
                    img_path = Path(image)
                    if not img_path.exists():
                        raise FileNotFoundError(f"Source image path '{image}' does not exist.")
                    pil_img = Image.open(img_path)
                elif isinstance(image, Image.Image):
                    pil_img = image
                else:
                    raise TypeError("image must be a PIL Image or a valid path/string to a file.")

                # Determine filename
                if not filename:
                    ts = int(time.time())
                    ext = "png" if fmt == "PNG" else "jpg"
                    filename = f"image_export_{ts}.{ext}"

                out_path = _EXPORTS_DIR / filename
                
                # PNG to JPEG conversion requires mode conversion if alpha/RGBA is present
                if fmt == "JPEG" and pil_img.mode in ("RGBA", "LA", "P"):
                    # Create white background pasteboard
                    bg = Image.new("RGB", pil_img.size, (255, 255, 255))
                    if pil_img.mode == "RGBA":
                        bg.paste(pil_img, mask=pil_img.split()[3]) # alpha mask
                    else:
                        bg.paste(pil_img.convert("RGBA"), mask=pil_img.convert("RGBA").split()[3])
                    pil_img = bg
                elif fmt == "JPEG" and pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

                # Save output
                if fmt == "JPEG":
                    pil_img.save(out_path, format="JPEG", quality=quality)
                else:
                    pil_img.save(out_path, format="PNG")

            except Exception as e:
                self._error_count += 1
                logger.error("ExportManager.export_image failed: %s", e, exc_info=True)
                return ServiceResult.fail(f"Image export failed: {e}", code="IMAGE_EXPORT_FAILED")

        logger.info("Exported image to %s (format: %s) in %.1fms", out_path.name, fmt, t.elapsed_ms)
        return ServiceResult.ok(
            data=str(out_path.resolve()),
            meta={"format": fmt, "latency_ms": t.elapsed_ms, "quality": quality},
            warnings=warnings,
            latency_ms=t.elapsed_ms,
        )

    # ── 2. Metadata Exporting ─────────────────────────────────────────────────

    def export_metadata(
        self,
        metadata: Dict[str, Any],
        format_type: str = "JSON",
        filename: Optional[str] = None,
    ) -> ServiceResult[str]:
        """Export prompt generation metadata parameters to JSON or CSV.

        Args:
            metadata:    Dictionary of parameters (prompt, seed, steps, etc.).
            format_type: Output format ("JSON" or "CSV").
            filename:    Optional filename on disk.

        Returns:
            ServiceResult wrapping the absolute path to the exported file.
        """
        self._call_count += 1
        fmt = format_type.upper().strip()

        if fmt not in ("JSON", "CSV"):
            self._error_count += 1
            return ServiceResult.validation_fail("format_type", "Format must be JSON or CSV.")

        with self._timer() as t:
            try:
                if not filename:
                    ts = int(time.time())
                    ext = "json" if fmt == "JSON" else "csv"
                    filename = f"metadata_{ts}.{ext}"

                out_path = _EXPORTS_DIR / filename

                if fmt == "JSON":
                    with open(out_path, "w", encoding="utf-8") as fh:
                        json.dump(metadata, fh, indent=2, ensure_ascii=False)
                else:
                    # Flat CSV key-value representation
                    with open(out_path, "w", newline="", encoding="utf-8") as fh:
                        writer = csv.writer(fh)
                        writer.writerow(["Parameter", "Value"])
                        for k, v in sorted(metadata.items()):
                            # Handle complex values
                            if isinstance(v, (list, dict)):
                                v = json.dumps(v, ensure_ascii=False)
                            writer.writerow([k, str(v)])

            except Exception as e:
                self._error_count += 1
                logger.error("ExportManager.export_metadata failed: %s", e, exc_info=True)
                return ServiceResult.fail(f"Metadata export failed: {e}", code="METADATA_EXPORT_FAILED")

        return ServiceResult.ok(
            data=str(out_path.resolve()),
            meta={"format": fmt, "latency_ms": t.elapsed_ms},
            latency_ms=t.elapsed_ms,
        )

    # ── 3. Recommendations Report Exporting ───────────────────────────────────

    def export_recommendation_report(
        self,
        report_data: Union[List[Dict[str, Any]], Dict[str, Any]],
        format_type: str = "JSON",
        filename: Optional[str] = None,
    ) -> ServiceResult[str]:
        """Export style or brand recommendation profiles.

        Args:
            report_data: List of recommendations or a recommendation profile dictionary.
            format_type: Format to export ("JSON", "CSV", or "MD" / "MARKDOWN").
            filename:    Optional custom filename.

        Returns:
            ServiceResult wrapping the absolute path to the report.
        """
        self._call_count += 1
        fmt = format_type.upper().strip()
        if fmt == "MARKDOWN":
            fmt = "MD"

        if fmt not in ("JSON", "CSV", "MD"):
            self._error_count += 1
            return ServiceResult.validation_fail("format_type", "Format must be JSON, CSV, or MD.")

        with self._timer() as t:
            try:
                # Standardize records list
                records: List[Dict[str, Any]] = []
                if isinstance(report_data, dict):
                    # Check if it has a list of items inside
                    items = report_data.get("recommendations") or report_data.get("styles") or report_data.get("brands")
                    if isinstance(items, list):
                        records = items
                    else:
                        records = [report_data]
                elif isinstance(report_data, list):
                    records = report_data

                if not filename:
                    ts = int(time.time())
                    ext = fmt.lower()
                    filename = f"recommendations_report_{ts}.{ext}"

                out_path = _EXPORTS_DIR / filename

                if fmt == "JSON":
                    with open(out_path, "w", encoding="utf-8") as fh:
                        json.dump(report_data, fh, indent=2, ensure_ascii=False)

                elif fmt == "CSV":
                    if not records:
                        raise ValueError("No records found to export to CSV.")
                    # Get flat headers from keys
                    headers = sorted(list({k for r in records for k in r.keys()}))
                    with open(out_path, "w", newline="", encoding="utf-8") as fh:
                        writer = csv.DictWriter(fh, fieldnames=headers, extrasaction="ignore")
                        writer.writeheader()
                        for r in records:
                            flat_row = {}
                            for h in headers:
                                val = r.get(h, "")
                                if isinstance(val, (list, dict)):
                                    val = json.dumps(val, ensure_ascii=False)
                                flat_row[h] = val
                            writer.writerow(flat_row)

                else:  # MD / Markdown report
                    with open(out_path, "w", encoding="utf-8") as fh:
                        fh.write("# 👗 AI Fashion Studio — Recommendation Report\n")
                        fh.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        fh.write("---\n\n")
                        
                        if isinstance(report_data, dict) and "user_id" in report_data:
                            fh.write(f"### 👤 User Preferences Profile: `{report_data['user_id']}`\n")
                            for k, v in report_data.items():
                                if k != "user_id" and not isinstance(v, (list, dict)):
                                    fh.write(f"- **{k.replace('_', ' ').title()}**: {v}\n")
                            fh.write("\n---\n\n")

                        fh.write("## 📝 Recommended Design Concepts\n\n")
                        for idx, r in enumerate(records, 1):
                            title = r.get("style") or r.get("brand") or r.get("name") or f"Option #{idx}"
                            fh.write(f"### {idx}. {title}\n")
                            
                            # Write details
                            for k, v in r.items():
                                if k in ("style", "brand", "name"):
                                    continue
                                if isinstance(v, list):
                                    items_str = ", ".join(f"`{item}`" for item in v)
                                    fh.write(f"- **{k.replace('_', ' ').title()}**: {items_str}\n")
                                elif isinstance(v, dict):
                                    fh.write(f"- **{k.replace('_', ' ').title()}**:\n")
                                    for subk, subv in v.items():
                                        fh.write(f"  - __{subk.replace('_', ' ').title()}__: {subv}\n")
                                else:
                                    fh.write(f"- **{k.replace('_', ' ').title()}**: {v}\n")
                            fh.write("\n")

            except Exception as e:
                self._error_count += 1
                logger.error("ExportManager.export_recommendation_report failed: %s", e, exc_info=True)
                return ServiceResult.fail(f"Recommendation report export failed: {e}", code="REPORT_EXPORT_FAILED")

        logger.info("Exported recommendations report to %s (format: %s)", out_path.name, fmt)
        return ServiceResult.ok(
            data=str(out_path.resolve()),
            meta={"format": fmt, "latency_ms": t.elapsed_ms},
            latency_ms=t.elapsed_ms,
        )

    # ── 4. Chat History Exporting ─────────────────────────────────────────────

    def export_chat_history(
        self,
        chat_history: Union[List[Tuple[str, str]], List[Dict[str, Any]]],
        format_type: str = "JSON",
        filename: Optional[str] = None,
    ) -> ServiceResult[str]:
        """Export chatbot dialog logs to JSON, CSV, or Markdown Q&A.

        Args:
            chat_history: List of conversation pairs [(User, Bot), ...] or list of messages dictionaries.
            format_type:  Output format ("JSON", "CSV", or "MD" / "MARKDOWN").
            filename:     Optional filename.

        Returns:
            ServiceResult wrapping the absolute path to the chat log.
        """
        self._call_count += 1
        fmt = format_type.upper().strip()
        if fmt == "MARKDOWN":
            fmt = "MD"

        if fmt not in ("JSON", "CSV", "MD"):
            self._error_count += 1
            return ServiceResult.validation_fail("format_type", "Format must be JSON, CSV, or MD.")

        with self._timer() as t:
            try:
                # Normalize history into a common list of dicts: [{"sender": "User/Assistant", "message": "..."}]
                normalized: List[Dict[str, Any]] = []
                for idx, item in enumerate(chat_history):
                    if isinstance(item, tuple) and len(item) == 2:
                        user_msg, bot_msg = item
                        if user_msg:
                            normalized.append({"index": idx * 2 + 1, "role": "user", "message": user_msg, "timestamp": ""})
                        if bot_msg:
                            normalized.append({"index": idx * 2 + 2, "role": "assistant", "message": bot_msg, "timestamp": ""})
                    elif isinstance(item, dict):
                        role = item.get("role") or item.get("sender") or "unknown"
                        msg = item.get("message") or item.get("content") or ""
                        ts = item.get("timestamp") or item.get("time") or ""
                        normalized.append({
                            "index": idx + 1,
                            "role": role,
                            "message": msg,
                            "timestamp": ts
                        })

                if not filename:
                    ts = int(time.time())
                    ext = fmt.lower()
                    filename = f"chat_history_{ts}.{ext}"

                out_path = _EXPORTS_DIR / filename

                if fmt == "JSON":
                    with open(out_path, "w", encoding="utf-8") as fh:
                        json.dump(chat_history, fh, indent=2, ensure_ascii=False)

                elif fmt == "CSV":
                    with open(out_path, "w", newline="", encoding="utf-8") as fh:
                        writer = csv.DictWriter(fh, fieldnames=["index", "role", "message", "timestamp"], extrasaction="ignore")
                        writer.writeheader()
                        for row in normalized:
                            writer.writerow(row)

                else:  # MD
                    with open(out_path, "w", encoding="utf-8") as fh:
                        fh.write("# 💬 AI Fashion Assistant — Conversation Transcript\n")
                        fh.write(f"Exported at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        fh.write("---\n\n")
                        
                        for row in normalized:
                            role = row["role"].upper()
                            msg = row["message"]
                            ts = row["timestamp"]
                            ts_str = f" *({ts})*" if ts else ""
                            
                            if role == "USER":
                                fh.write(f"🗣️ **USER**{ts_str}:\n{msg}\n\n")
                            else:
                                fh.write(f"🤖 **ASSISTANT**{ts_str}:\n{msg}\n\n")
                                fh.write("---\n\n")

            except Exception as e:
                self._error_count += 1
                logger.error("ExportManager.export_chat_history failed: %s", e, exc_info=True)
                return ServiceResult.fail(f"Chat history export failed: {e}", code="CHAT_EXPORT_FAILED")

        logger.info("Exported chat history to %s (format: %s)", out_path.name, fmt)
        return ServiceResult.ok(
            data=str(out_path.resolve()),
            meta={"format": fmt, "latency_ms": t.elapsed_ms},
            latency_ms=t.elapsed_ms,
        )

    # ── 5. Batch ZIP Archiver ─────────────────────────────────────────────────

    def export_batch_zip(
        self,
        records: List[Dict[str, Any]],
        archive_name: str = "fashion_studio_export",
    ) -> ServiceResult[str]:
        """Compress multiple generated concepts, metadata, or reports into a single ZIP.

        Args:
            records:      List of record dictionaries. Each dictionary must contain:
                          - ``type``: Type of asset to pack ("image", "metadata", "report", "chat").
                          - ``source``: The source object (Image, dictionary, path string).
                          - ``filename``: Filename inside the zip archive (e.g. "design.png").
            archive_name: Target name of the ZIP file (will be appended with timestamp if no .zip).

        Returns:
            ServiceResult wrapping the absolute path to the generated ZIP file.
        """
        self._call_count += 1
        
        if not records:
            return ServiceResult.validation_fail("records", "Records list must not be empty.")

        # Clean archive name
        archive_name = archive_name.strip()
        if not archive_name.lower().endswith(".zip"):
            ts = int(time.time())
            archive_name = f"{archive_name}_{ts}.zip"

        out_path = _EXPORTS_DIR / archive_name
        warnings: List[str] = []

        with self._timer() as t:
            try:
                # Create temporary directory inside workspace exports directory to stage files if needed
                stage_dir = _EXPORTS_DIR / f"temp_stage_{int(time.time())}"
                stage_dir.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for idx, r in enumerate(records):
                        r_type = r.get("type", "").lower()
                        source = r.get("source")
                        fname = r.get("filename") or f"asset_{idx}"

                        if not source:
                            warnings.append(f"Record #{idx} skipped: Missing 'source' attribute.")
                            continue

                        # ── Handle Image ──────────────────────────────────────
                        if r_type == "image":
                            # Check if source is on-disk or PIL Image
                            if isinstance(source, (str, Path)):
                                p = Path(source)
                                if p.exists():
                                    zipf.write(p, arcname=fname)
                                else:
                                    warnings.append(f"Image path '{source}' not found. Skipped.")
                            elif isinstance(source, Image.Image):
                                temp_img = stage_dir / fname
                                source.save(temp_img)
                                zipf.write(temp_img, arcname=fname)
                            else:
                                warnings.append(f"Invalid image source type '{type(source)}'. Skipped.")

                        # ── Handle Metadata / Reports / Chat Logs ─────────────
                        elif r_type in ("metadata", "report", "chat"):
                            temp_file = stage_dir / fname
                            
                            # Decide how to dump based on source type and file extension
                            if fname.lower().endswith(".json"):
                                with open(temp_file, "w", encoding="utf-8") as fh:
                                    json.dump(source, fh, indent=2, ensure_ascii=False)
                            elif fname.lower().endswith(".csv") and isinstance(source, dict):
                                with open(temp_file, "w", newline="", encoding="utf-8") as fh:
                                    writer = csv.writer(fh)
                                    writer.writerow(["Key", "Value"])
                                    for k, v in sorted(source.items()):
                                        writer.writerow([k, str(v)])
                            elif isinstance(source, str):
                                with open(temp_file, "w", encoding="utf-8") as fh:
                                    fh.write(source)
                            else:
                                # String representation fallback
                                with open(temp_file, "w", encoding="utf-8") as fh:
                                    fh.write(str(source))
                                    
                            zipf.write(temp_file, arcname=fname)
                        else:
                            # Direct file write if source is an existing file path
                            if isinstance(source, (str, Path)):
                                p = Path(source)
                                if p.exists():
                                    zipf.write(p, arcname=fname)
                                else:
                                    warnings.append(f"File path '{source}' not found. Skipped.")
                            else:
                                warnings.append(f"Unknown type '{r_type}' and non-path source. Skipped.")

                # Cleanup stage folder
                if stage_dir.exists():
                    for f in stage_dir.glob("*"):
                        try:
                            f.unlink()
                        except Exception:
                            pass
                    stage_dir.rmdir()

            except Exception as e:
                self._error_count += 1
                logger.error("ExportManager.export_batch_zip failed: %s", e, exc_info=True)
                return ServiceResult.fail(f"ZIP archiving failed: {e}", code="ZIP_EXPORT_FAILED")

        logger.info("Successfully packaged %d concepts into ZIP archive: %s", len(records), out_path.name)
        return ServiceResult.ok(
            data=str(out_path.resolve()),
            meta={"latency_ms": t.elapsed_ms, "archive_name": out_path.name},
            warnings=warnings,
            latency_ms=t.elapsed_ms,
        )

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """Lightweight status probe to check directory permissions and status."""
        try:
            # Check write permission to exports folder
            test_file = _EXPORTS_DIR / ".permissions_probe"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            return {
                "status":    "ok",
                "mock_mode": self.mock_mode,
                "backend":   "local_fs",
                "message":   f"Exports folder {_EXPORTS_DIR} is fully readable/writable.",
            }
        except Exception as e:
            return {
                "status":    "error",
                "mock_mode": self.mock_mode,
                "backend":   "local_fs",
                "message":   f"Directory write permission check failed: {e}",
            }

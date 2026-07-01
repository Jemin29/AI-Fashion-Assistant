#!/usr/bin/env python3
"""
scripts/restore.py
==================
Restore script for the AI Fashion Assistant platform.
Performs SHA-256 integrity verification, extracts ZIP archives or copies folders,
and features a complete rollback safety mechanism to prevent workspace corruption.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Mapping of component names to list of relative file/directory paths
COMPONENT_MAP: Dict[str, List[str]] = {
    "models": [
        "outputs/trainer",
        "outputs/embeddings/cache",
        "outputs/loras",
    ],
    "chromadb": [
        "outputs/chroma_db",
        "outputs/vector_db/faiss_index",
    ],
    "outputs": [
        "outputs/generated",
        "outputs/portfolio",
        "outputs/recommendations",
        "outputs/sketches",
    ],
    "configs": [
        "configs",
        "week7/backend/configs",
        "week8/configs",
        ".env",
        "week7/backend/configs/.env",
        "week8/configs/.env.production",
        "requirements.txt",
        "docker-compose.yml",
        "Dockerfile",
    ],
    "history": [
        "outputs/history.jsonl",
        "week6/outputs/history/history.jsonl",
        "outputs/experiments/experiments.db",
        "fashion_assistant.db",
        "week7/backend/fashion_assistant.db",
    ],
}


def calculate_sha256(filepath: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_metadata(backup_path: Path) -> Dict[str, any]:
    """Locate and load companion metadata for the backup path."""
    meta_path = Path(str(backup_path).replace(".zip", "") + "_meta.json")
    if not meta_path.exists():
        # Try looking in the parent directory
        meta_path = backup_path.parent / f"{backup_path.stem}_meta.json"

    if not meta_path.exists():
        print(f"[!] Warning: Companion metadata file not found at: {meta_path}")
        return {}

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[!] Error reading metadata file: {exc}")
        return {}


def is_file_in_components(rel_path: Path, components: List[str]) -> bool:
    """Determine if a file path belongs to any of the selected components."""
    path_str = rel_path.as_posix()
    for comp in components:
        for pattern in COMPONENT_MAP[comp]:
            # Matches exact file or anything within the folder
            if path_str == pattern or path_str.startswith(pattern + "/"):
                return True
    return False


def run_restore(
    workspace_root: Path,
    backup_path: Path,
    selected_components: List[str],
    force: bool = False
) -> bool:
    """Execute the restore pipeline, verifying integrity and utilizing safety rollback."""
    if not backup_path.exists():
        print(f"[ERROR] Backup source path does not exist: {backup_path}")
        return False

    is_zip = backup_path.is_file() and backup_path.suffix.lower() == ".zip"
    meta = load_metadata(backup_path)

    # 1. Verification phase
    if is_zip and meta:
        print("[*] Verifying backup archive integrity...")
        expected_sha = meta.get("archive_sha256", "")
        if expected_sha:
            actual_sha = calculate_sha256(backup_path)
            if actual_sha != expected_sha:
                print(f"[ERROR] SHA-256 Checksum Verification Failed!")
                print(f"  Expected: {expected_sha}")
                print(f"  Actual:   {actual_sha}")
                return False
            print("[+] Checksum validation passed successfully.")
    
    # Identify components inside the backup
    backup_components = list(meta.get("components", {}).keys()) if meta else selected_components
    components_to_restore = [c for c in selected_components if c in backup_components or not meta]

    if not components_to_restore:
        print("[ERROR] No overlapping components found between selection and backup archive.")
        return False

    print(f"[*] Target restore directory: {workspace_root.resolve()}")
    print(f"[*] Restoring components: {', '.join(components_to_restore)}")

    if not force:
        confirm = input("[?] WARNING: This restore will overwrite existing files. Continue? [y/N]: ")
        if confirm.strip().lower() not in ("y", "yes"):
            print("[*] Restore aborted by user.")
            return False

    # 2. Safety Rollback preparation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backups_dir = workspace_root / "backups"
    backups_dir.mkdir(exist_ok=True)
    rollback_zip_path = backups_dir / f"rollback_temp_{timestamp}.zip"
    
    files_to_restore: List[str] = []

    if is_zip:
        with zipfile.ZipFile(backup_path, "r") as zipf:
            files_to_restore = [
                name for name in zipf.namelist()
                if is_file_in_components(Path(name), components_to_restore)
            ]
    else:
        # For folders, crawl relative files
        for root, _, filenames in os.walk(backup_path):
            root_path = Path(root)
            for filename in filenames:
                file_path = root_path / filename
                rel = file_path.relative_to(backup_path)
                if is_file_in_components(rel, components_to_restore):
                    files_to_restore.append(rel.as_posix())

    if not files_to_restore:
        print("[!] No files match the chosen components in the backup archive.")
        return True

    print(f"[*] Creating rollback safety backup of {len(files_to_restore)} potential target files...")
    rollback_files_count = 0
    try:
        with zipfile.ZipFile(rollback_zip_path, "w", zipfile.ZIP_DEFLATED) as rzipf:
            for rel_path_str in files_to_restore:
                target_file = workspace_root / rel_path_str
                if target_file.exists() and target_file.is_file():
                    rzipf.write(target_file, rel_path_str)
                    rollback_files_count += 1
        print(f"[+] Rollback safety backup saved to {rollback_zip_path.name} ({rollback_files_count} files stored).")
    except Exception as exc:
        print(f"[ERROR] Failed to compile rollback safety backup: {exc}")
        if rollback_zip_path.exists():
            rollback_zip_path.unlink()
        return False

    # 3. Extraction/Copy phase with transactional try-except rollback
    files_modified_or_created: List[Path] = []
    restore_success = True

    try:
        if is_zip:
            print(f"[*] Extracting files from ZIP...")
            with zipfile.ZipFile(backup_path, "r") as zipf:
                for name in files_to_restore:
                    target_file = workspace_root / name
                    # Track for cleanups if we crash
                    files_modified_or_created.append(target_file)
                    
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    # Extract single file
                    with zipf.open(name) as source, open(target_file, "wb") as target:
                        shutil.copyfileobj(source, target)
        else:
            print(f"[*] Copying files from folder...")
            for name in files_to_restore:
                src_file = backup_path / name
                target_file = workspace_root / name
                files_modified_or_created.append(target_file)
                
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, target_file)

        print("[+] Files extracted successfully.")

    except Exception as exc:
        print(f"[CRITICAL] Error occurred during restore extraction: {exc}")
        restore_success = False

    # 4. Rollback execution if failed, or cleanup on success
    if not restore_success:
        print("[*] Initiating rollback procedure...")
        # Delete any files created or modified during the failed restore
        for path in files_modified_or_created:
            try:
                if path.exists() and path.is_file():
                    path.unlink()
            except Exception:
                pass

        if rollback_files_count > 0 and rollback_zip_path.exists():
            try:
                print(f"[*] Restoring {rollback_files_count} original files from rollback archive...")
                with zipfile.ZipFile(rollback_zip_path, "r") as rzipf:
                    rzipf.extractall(workspace_root)
                print("[+] Rollback successfully completed. Workspace restored to original state.")
            except Exception as rollback_exc:
                print(f"[FATAL] Failed to perform rollback from safety backup! Workspace might be inconsistent: {rollback_exc}")
        else:
            print("[*] No original files existed to roll back. Workspace cleaned.")

        # Cleanup rollback zip
        if rollback_zip_path.exists():
            rollback_zip_path.unlink()
        return False
    else:
        # Success cleanup
        if rollback_zip_path.exists():
            try:
                rollback_zip_path.unlink()
                print("[*] Cleaned up temporary rollback safety archives.")
            except Exception:
                pass
        return True


def main() -> None:
    workspace_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Restore utility for the AI Fashion Assistant.")
    parser.add_argument(
        "backup_path",
        type=str,
        help="Path to the backup ZIP file or raw backup directory to restore from."
    )
    parser.add_argument(
        "--components",
        type=str,
        default="all",
        help="Comma-separated list of components to restore (models,chromadb,outputs,configs,history) or 'all'."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Execute restore operations directly without interactive confirmations."
    )

    args = parser.parse_args()

    backup_path = Path(args.backup_path)
    if not backup_path.is_absolute():
        backup_path = workspace_root / backup_path

    # Parse components
    if args.components.lower() == "all":
        components = list(COMPONENT_MAP.keys())
    else:
        components = [c.strip().lower() for c in args.components.split(",")]
        # Validate selection
        invalid = [c for c in components if c not in COMPONENT_MAP]
        if invalid:
            parser.error(f"Invalid component(s): {', '.join(invalid)}. Supported: {list(COMPONENT_MAP.keys())}")

    success = run_restore(
        workspace_root=workspace_root,
        backup_path=backup_path,
        selected_components=components,
        force=args.force
    )

    if success:
        print("\n[SUCCESS] Restore operation completed successfully.")
        sys.exit(0)
    else:
        print("\n[FAILURE] Restore operation failed or was aborted.")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/backup.py
=================
Backup script for the AI Fashion Assistant platform.
Supports ZIP compression, component-level selection, metadata generation with
SHA-256 checksums, and directory/file exclusions.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Exclude patterns to ignore during backup
EXCLUDE_PATTERNS: Set[str] = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".coverage",
    "htmlcov",
    ".venv",
    "venv",
    "env",
    "backups",  # avoid backing up the backup directory recursively
}

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


def should_exclude(path: Path) -> bool:
    """Check if the given path matches any exclude patterns."""
    for part in path.parts:
        if part in EXCLUDE_PATTERNS:
            return True
        if part.endswith(".pyc") or part.endswith(".pyo"):
            return True
    return False


def get_file_list(workspace_root: Path, relative_paths: List[str]) -> List[Path]:
    """Gather all files within specified relative paths, excluding matching patterns."""
    files = []
    seen = set()
    for rel_path_str in relative_paths:
        full_path = workspace_root / rel_path_str
        if not full_path.exists():
            continue

        if full_path.is_file():
            resolved = full_path.resolve()
            if resolved not in seen:
                if not should_exclude(full_path.relative_to(workspace_root)):
                    files.append(full_path)
                    seen.add(resolved)
        elif full_path.is_dir():
            for root, _, filenames in os.walk(full_path):
                root_path = Path(root)
                for filename in filenames:
                    file_path = root_path / filename
                    resolved = file_path.resolve()
                    if resolved not in seen:
                        if not should_exclude(file_path.relative_to(workspace_root)):
                            files.append(file_path)
                            seen.add(resolved)
    return files


def calculate_sha256(filepath: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def run_backup(
    workspace_root: Path,
    backup_dir: Path,
    selected_components: List[str],
    compress: bool = True
) -> Tuple[Path, Path]:
    """Execute the backup routine and return paths to archive/folder and metadata JSON."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"fashion_backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting backup: {backup_name}")
    print(f"[*] Target backup directory: {backup_dir.resolve()}")
    print(f"[*] Selected components: {', '.join(selected_components)}")

    component_stats: Dict[str, Dict[str, any]] = {}
    total_files = 0
    total_bytes = 0

    # 1. Gather all files to back up and calculate stats
    all_files_to_backup: List[Tuple[str, Path]] = []  # List of (component_name, file_path)

    for comp in selected_components:
        paths = COMPONENT_MAP[comp]
        comp_files = get_file_list(workspace_root, paths)
        
        comp_size = sum(f.stat().st_size for f in comp_files)
        comp_count = len(comp_files)
        
        component_stats[comp] = {
            "file_count": comp_count,
            "size_bytes": comp_size
        }
        
        total_files += comp_count
        total_bytes += comp_size

        for f in comp_files:
            all_files_to_backup.append((comp, f))

    if not all_files_to_backup:
        print("[!] Warning: No files found to backup for the selected components!")

    # 2. Perform copy or compression
    if compress:
        zip_path = backup_dir / f"{backup_name}.zip"
        print(f"[*] Packaging {total_files} files into ZIP archive...")
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for _, file_path in all_files_to_backup:
                # Add file with relative path corresponding to workspace root
                arcname = file_path.relative_to(workspace_root)
                zipf.write(file_path, arcname)

        archive_size = zip_path.stat().st_size
        archive_sha = calculate_sha256(zip_path)
        dest_path = zip_path
        
        print(f"[+] Backup ZIP created successfully: {zip_path.name} ({archive_size / (1024*1024):.2f} MB)")
    else:
        # Create uncompressed directory copy
        folder_path = backup_dir / backup_name
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"[*] Copying {total_files} files to folder structure...")
        
        for _, file_path in all_files_to_backup:
            rel_path = file_path.relative_to(workspace_root)
            target_file_path = folder_path / rel_path
            target_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target_file_path)

        archive_size = sum(
            f.stat().st_size for root, _, filenames in os.walk(folder_path) for f in [Path(root) / fn]
        )
        archive_sha = ""  # No hash for raw folders
        dest_path = folder_path
        
        print(f"[+] Backup folder created successfully: {folder_path.name}")

    # 3. Create metadata file
    meta_path = backup_dir / f"{backup_name}_meta.json"
    metadata = {
        "backup_name": backup_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "compress": compress,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "archive_size_bytes": archive_size,
        "archive_sha256": archive_sha,
        "components": component_stats,
        "system": {
            "os": os.name,
            "platform": datetime.datetime.now().strftime("%A, %d %B %Y")
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Backup metadata saved: {meta_path.name}")

    return dest_path, meta_path


def main() -> None:
    workspace_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Backup utility for the AI Fashion Assistant.")
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="backups",
        help="Target folder for backup archives (relative to workspace root or absolute)."
    )
    parser.add_argument(
        "--components",
        type=str,
        default="all",
        help="Comma-separated list of components to backup (models,chromadb,outputs,configs,history) or 'all'."
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Skip ZIP compression and copy files directly to a folder."
    )

    args = parser.parse_args()

    # Determine backup folder path
    backup_path = Path(args.backup_dir)
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

    dest, meta = run_backup(
        workspace_root=workspace_root,
        backup_dir=backup_path,
        selected_components=components,
        compress=not args.no_compress
    )
    print(f"\n[SUCCESS] Backup execution complete.")
    print(f"Archive/Folder: {dest.resolve()}")
    print(f"Metadata File:  {meta.resolve()}")


if __name__ == "__main__":
    main()

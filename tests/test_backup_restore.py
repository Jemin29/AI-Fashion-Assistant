from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
import pytest

from scripts.backup import run_backup, COMPONENT_MAP, calculate_sha256
from scripts.restore import run_restore


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory mimicking the project structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create some dummy files corresponding to the COMPONENT_MAP structure
        # Configs
        configs_dir = root / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        with open(configs_dir / "settings.yaml", "w") as f:
            f.write("project_name: test_fashion_assistant")
            
        w7_configs_dir = root / "week7/backend/configs"
        w7_configs_dir.mkdir(parents=True, exist_ok=True)
        with open(w7_configs_dir / "config.py", "w") as f:
            f.write("sqlite_url = 'sqlite://'")
            
        # ChromaDB
        chroma_dir = root / "outputs/chroma_db"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        with open(chroma_dir / "chroma.sqlite3", "w") as f:
            f.write("mock_database_data")
            
        # Outputs
        generated_dir = root / "outputs/generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        with open(generated_dir / "image.png", "w") as f:
            f.write("png_mock_binary_data")
            
        # History
        with open(root / "outputs/history.jsonl", "w") as f:
            f.write('{"event": "test_generation"}\n')

        yield root


def test_full_backup_and_restore(temp_workspace):
    """Verify that a full backup and restore cycle completes successfully."""
    workspace = temp_workspace
    backup_dir = workspace / "backups"
    
    # Run full backup
    components = ["configs", "chromadb", "outputs", "history"]
    dest_path, meta_path = run_backup(
        workspace_root=workspace,
        backup_dir=backup_dir,
        selected_components=components,
        compress=True
    )
    
    assert dest_path.exists()
    assert dest_path.suffix == ".zip"
    assert meta_path.exists()
    
    # Read metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["total_files"] == 5
    assert meta["compress"] is True
    assert "configs" in meta["components"]
    assert "chromadb" in meta["components"]
    
    # Verify the hash matches
    assert calculate_sha256(dest_path) == meta["archive_sha256"]
    
    # Modify files in workspace to simulate user changes or corruption
    settings_file = workspace / "configs/settings.yaml"
    with open(settings_file, "w") as f:
        f.write("project_name: corrupted_value")
        
    history_file = workspace / "outputs/history.jsonl"
    if history_file.exists():
        history_file.unlink()
        
    # Run restore
    success = run_restore(
        workspace_root=workspace,
        backup_path=dest_path,
        selected_components=components,
        force=True
    )
    
    assert success is True
    
    # Verify original file content is restored
    with open(settings_file, "r") as f:
        content = f.read()
    assert content == "project_name: test_fashion_assistant"
    
    assert history_file.exists()
    with open(history_file, "r") as f:
        history_content = f.read()
    assert "test_generation" in history_content


def test_selective_backup_and_restore(temp_workspace):
    """Verify selective component backups (e.g. only configs)."""
    workspace = temp_workspace
    backup_dir = workspace / "backups"
    
    # Run selective backup of configs only
    dest_path, meta_path = run_backup(
        workspace_root=workspace,
        backup_dir=backup_dir,
        selected_components=["configs"],
        compress=True
    )
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    assert "configs" in meta["components"]
    assert "chromadb" not in meta["components"]
    assert meta["total_files"] == 2  # settings.yaml + config.py
    
    # Modify configs and outputs
    settings_file = workspace / "configs/settings.yaml"
    with open(settings_file, "w") as f:
        f.write("changed_config")
        
    chroma_file = workspace / "outputs/chroma_db/chroma.sqlite3"
    with open(chroma_file, "w") as f:
        f.write("changed_database")
        
    # Restore configs component
    success = run_restore(
        workspace_root=workspace,
        backup_path=dest_path,
        selected_components=["configs"],
        force=True
    )
    assert success is True
    
    # Configs should be restored
    with open(settings_file, "r") as f:
        assert f.read() == "project_name: test_fashion_assistant"
        
    # Chroma DB should NOT be restored (since it wasn't backed up in this file)
    with open(chroma_file, "r") as f:
        assert f.read() == "changed_database"


def test_restore_checksum_mismatch(temp_workspace):
    """Confirm that restore is aborted if the archive checksum mismatches the metadata."""
    workspace = temp_workspace
    backup_dir = workspace / "backups"
    
    dest_path, meta_path = run_backup(
        workspace_root=workspace,
        backup_dir=backup_dir,
        selected_components=["configs"],
        compress=True
    )
    
    # Modify ZIP archive bytes to corrupt it and break checksum
    with open(dest_path, "ab") as f:
        f.write(b"corrupting_extra_bytes")
        
    # Attempt restore
    success = run_restore(
        workspace_root=workspace,
        backup_path=dest_path,
        selected_components=["configs"],
        force=True
    )
    
    # Should fail due to checksum mismatch
    assert success is False


def test_rollback_safety_mechanism(temp_workspace):
    """Test that if the restore process fails mid-way, original files are rolled back."""
    workspace = temp_workspace
    backup_dir = workspace / "backups"
    
    # Make backup
    dest_path, meta_path = run_backup(
        workspace_root=workspace,
        backup_dir=backup_dir,
        selected_components=["configs"],
        compress=True
    )
    
    # Check original value
    settings_file = workspace / "configs/settings.yaml"
    
    # Corrupt the ZIP internally so it raises BadZipFile/error halfway during extraction
    # We can write a truncated zip file over it
    with open(dest_path, "wb") as f:
        f.write(b"invalid_truncated_zip_format_data")
        
    # Change settings file in workspace
    with open(settings_file, "w") as f:
        f.write("modified_workspace_settings")
        
    # Attempt restore. This will fail during unzip extraction.
    success = run_restore(
        workspace_root=workspace,
        backup_path=dest_path,
        selected_components=["configs"],
        force=True
    )
    
    assert success is False
    
    # The rollback safety mechanism should have triggered, recovering the original file
    with open(settings_file, "r") as f:
        restored_val = f.read()
    assert restored_val == "modified_workspace_settings"

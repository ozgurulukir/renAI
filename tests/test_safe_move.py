import sys
sys.path.insert(0, "src")

#!/usr/bin/env python3
"""Test script for safe_move_file function"""

import tempfile
import shutil
from pathlib import Path
from renai.file_ops import safe_move_file


def test_safe_move_basic():
    """Test basic file moving functionality"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create source file
        source_file = tmp_path / "test.txt"
        source_file.write_text("Hello World")
        
        # Test basic move
        dest_file = tmp_path / "moved.txt"
        result = safe_move_file(source_file, dest_file, "test")
        
        assert result == dest_file
        assert dest_file.exists()
        assert not source_file.exists()
        print("PASS: Basic file move test passed")


def test_safe_move_duplicate_same_content():
    """Test moving file when destination exists with same content"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create identical files
        source_file = tmp_path / "source.txt"
        dest_file = tmp_path / "dest.txt"
        
        content = "Same content"
        source_file.write_text(content)
        dest_file.write_text(content)
        
        # Should skip moving since content is identical
        result = safe_move_file(source_file, dest_file, "test")
        
        assert result == dest_file
        assert dest_file.exists()
        assert source_file.exists()  # Source should still exist
        print("PASS: Duplicate same content test passed")


def test_safe_move_duplicate_different_content():
    """Test moving file when destination exists with different content"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create files with different content
        source_file = tmp_path / "source.txt"
        dest_file = tmp_path / "dest.txt"
        
        source_file.write_text("Source content")
        dest_file.write_text("Destination content")
        
        # Should create new file with UUID suffix
        result = safe_move_file(source_file, dest_file, "test")
        
        assert result != dest_file
        assert "_" in result.name
        assert result.exists()
        assert not source_file.exists()
        assert dest_file.exists()  # Original dest should still exist
        print("PASS: Duplicate different content test passed")


def test_safe_move_missing_source():
    """Test error handling when source file doesn't exist"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        source_file = tmp_path / "nonexistent.txt"
        dest_file = tmp_path / "dest.txt"
        
        try:
            safe_move_file(source_file, dest_file, "test")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            print("PASS: Missing source file test passed")


def test_safe_move_thread_safety():
    """Test that file operations are thread-safe"""
    import threading
    import time
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create multiple source files
        source_files = [tmp_path / f"source_{i}.txt" for i in range(5)]
        for sf in source_files:
            sf.write_text(f"Content for {sf.name}")
        
        # Single destination file
        dest_file = tmp_path / "dest.txt"
        dest_file.write_text("Initial content")
        
        results = []
        errors = []
        
        def move_file(source_path):
            try:
                result = safe_move_file(source_path, dest_file, "thread_test")
                results.append((source_path, result))
            except Exception as e:
                errors.append((source_path, str(e)))
        
        # Start multiple threads
        threads = []
        for source_file in source_files:
            thread = threading.Thread(target=move_file, args=(source_file,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 5
        assert len(errors) == 0
        
        # Should have one original dest and 4 new files with UUIDs
        moved_files = [r[1] for r in results]
        unique_files = set(moved_files)
        assert len(unique_files) == 5  # All should be unique
        
        print("PASS: Thread safety test passed")


if __name__ == "__main__":
    print("Running safe_move_file tests...\n")
    
    test_safe_move_basic()
    test_safe_move_duplicate_same_content()
    test_safe_move_duplicate_different_content()
    test_safe_move_missing_source()
    test_safe_move_thread_safety()
    
    print("\nSUCCESS: All tests passed!")
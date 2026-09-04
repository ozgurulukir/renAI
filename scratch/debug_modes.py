import asyncio
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add src to sys.path to import renai
sys.path.append("src")

from renai.models.schemas import BookMetadata
from renai.processors.book import BookProcessor


async def reproduce():
    test_dir = Path("reproduce_issue")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    # Create a mock book
    book_file = test_dir / "old_book.pdf"
    book_file.write_bytes(b"pdf content")

    mock_ocr = MagicMock()
    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    metadata = BookMetadata(
        title="Physics 101",
        author="Einstein",
        category="Science",
        year=1905,
        language="en",
        subtitle=None,
        edition=None,
    )
    mock_llm.get_book_metadata = AsyncMock(return_value=metadata)

    processor = BookProcessor(mock_ocr, mock_llm)
    # Mock extractor to return something
    processor.text_extractor = MagicMock()
    processor.text_extractor.extract = AsyncMock(return_value="Physics stuff")

    print("\n--- TEST 1: Organize Mode (First Run) ---")
    res1 = await processor.process_file(book_file, mode="organize")
    print(f"Result 1: {res1.file_path}")

    assert res1.file_path is not None

    print("\n--- TEST 2: Organize Mode (Second Run - Double Nesting Bug) ---")
    # Simulate processing the already renamed/moved file
    res2 = await processor.process_file(res1.file_path, mode="organize")
    print(f"Result 2: {res2.file_path}")

    if "Science/Science" in str(res2.file_path):
        print("[BUG] BUG CONFIRMED: Double nesting occurred!")
    else:
        print("[OK] No double nesting.")

    print("\n--- TEST 3: Organize Mode (Correct Name, Wrong Place - Skip Bug) ---")
    # Create a file with the correct name but in the wrong folder
    correct_name = "Physics 101 -- Einstein -- (1905).pdf"
    wrong_place = test_dir / correct_name
    wrong_place.write_bytes(b"content")

    res3 = await processor.process_file(wrong_place, mode="organize")
    print(f"Result 3: {res3.file_path}")

    if res3.file_path == wrong_place:
        print("[BUG] BUG CONFIRMED: Skipped move because name matches!")
    else:
        print("[OK] Moved to correct category folder.")

    shutil.rmtree(test_dir)


if __name__ == "__main__":
    asyncio.run(reproduce())

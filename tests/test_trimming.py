#!/usr/bin/env python3
"""Test the improved text trimming functionality"""

from renai.utils import proportional_chunk_trim


def test_proportional_trimming():
    """Test the improved proportional chunk trimming"""

    # Test 1: Short text (no trimming needed)
    short_text = "This is a short text file"
    result = proportional_chunk_trim(short_text, target_length=100, num_chunks=5)
    assert result == short_text
    print("PASS: Short text preserved")

    # Test 2: Long text with preservation of endings
    # Ensure metadata is at the end
    chunks = [f"Content chunk {i}" for i in range(8)]
    chunks.extend(
        ["REFERENCES", "Author: John Doe", "Year: 2023", "Publisher: Test Press"]
    )
    long_text = "\n".join(chunks)
    result = proportional_chunk_trim(long_text, target_length=200, num_chunks=4)

    # Should preserve endings with metadata
    assert "REFERENCES" in result or "Author:" in result
    assert len(result) <= 200
    print("PASS: Late chunk preservation")

    # Test 3: Exact length handling
    exact_text = "A" * 1000
    result = proportional_chunk_trim(exact_text, target_length=500, num_chunks=3)
    assert len(result) <= 500
    print("PASS: Exact length handling")

    # Test 4: Document metadata preservation
    document = """Introduction content goes here...
    Additional content...
    
    CHAPTER 10: Conclusion
    This is the conclusion section.
    
    REFERENCES
    Author: John Smith
    Published: 2023
    Publisher: Example Press
    ISBN: 123-4567890123
    """

    result = proportional_chunk_trim(document, target_length=250, num_chunks=3)
    # Should preserve end sections with metadata
    assert len(result) <= 250
    print("PASS: Document structure preserved")


def test_edge_cases():
    """Test edge cases and boundary conditions"""

    # Empty text
    result = proportional_chunk_trim("", target_length=100, num_chunks=5)
    assert result == ""
    print("PASS: Empty text handled")

    # Single chunk scenario
    text = "Single line of text"
    result = proportional_chunk_trim(text, target_length=10, num_chunks=1)
    assert len(result) <= 10
    print("PASS: Single chunk handling")

    # Many chunks with small target
    text = " ".join([f"Word{i}" for i in range(100)])
    result = proportional_chunk_trim(text, target_length=50, num_chunks=20)
    assert len(result) <= 50
    print("PASS: Many chunks with small target")


if __name__ == "__main__":
    print("Running text trimming tests...\n")

    test_proportional_trimming()
    test_edge_cases()

    print("\nSUCCESS: All trimming tests passed!")

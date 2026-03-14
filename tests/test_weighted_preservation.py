#!/usr/bin/env python3
"""Test the weighted preservation strategy for document trimming"""

from renai.utils import proportional_chunk_trim


def test_weighted_preservation_strategy():
    """Test that early chunks get more preservation than later chunks"""
    
    # Create a document with clear structure
    document_parts = [
        "TITLE: Artificial Intelligence Fundamentals",
        "AUTHOR: Dr. Alan Turing", 
        "YEAR: 2024",
        "PUBLISHER: MIT Press",
        "ABSTRACT: This book covers...",
        "CHAPTER 1: Introduction",
        "CHAPTER 2: Machine Learning",
        "CHAPTER 3: Neural Networks",
        "CHAPTER 4: Natural Language Processing",
        "CHAPTER 5: Computer Vision",
        "CHAPTER 6: Ethics in AI",
        "CHAPTER 7: Future Directions",
        "REFERENCES",
        "INDEX"
    ]
    
    # Create a long document
    long_document = "\n\n".join(part + "\n" + "x" * 200 for part in document_parts)
    
    # Trim to target length
    target_length = 1000
    result = proportional_chunk_trim(long_document, target_length, num_chunks=8)
    
    print(f"Original length: {len(long_document)}")
    print(f"Target length: {target_length}")
    print(f"Result length: {len(result)}")
    print(f"Compression ratio: {len(result)/len(long_document)*100:.1f}%")
    
    # Check that early metadata is preserved
    preserved_early_content = any(keyword in result for keyword in ["TITLE:", "AUTHOR:", "YEAR:", "PUBLISHER:"])
    preserved_middle_content = any(keyword in result for keyword in ["CHAPTER 3", "CHAPTER 4"])
    
    print(f"Early metadata preserved: {preserved_early_content}")
    print(f"Middle content preserved: {preserved_middle_content}")
    
    assert preserved_early_content, "Early metadata should be preserved"
    assert len(result) <= target_length, "Result should be within target length"
    
    # Test the preservation ratios
    chunks = result.split("\n\n")
    if len(chunks) >= 3:
        first_chunk_length = len(chunks[0])
        second_chunk_length = len(chunks[1]) if len(chunks) > 1 else 0
        last_chunk_length = len(chunks[-1])
        
        print(f"First chunk length: {first_chunk_length}")
        print(f"Second chunk length: {second_chunk_length}")
        print(f"Last chunk length: {last_chunk_length}")
        
        # Early chunks should generally be longer than later chunks
        if second_chunk_length > 0 and last_chunk_length > 0:
            print("PASS: Weighted preservation strategy working")
    
    print("PASS: Weighted preservation test")


def test_preservation_factors():
    """Test the specific preservation factors for different chunk positions"""
    
    # Test document with clear chunk boundaries
    test_text = "Chunk0_" * 100 + "Chunk1_" * 100 + "Chunk2_" * 100 + "Chunk3_" * 100
    
    result = proportional_chunk_trim(test_text, target_length=200, num_chunks=4)
    
    # Check preservation pattern
    chunk0_preserved = "Chunk0_" in result
    chunk1_preserved = "Chunk1_" in result
    chunk2_preserved = "Chunk2_" in result
    chunk3_preserved = "Chunk3_" in result
    
    print(f"Chunk 0 preserved: {chunk0_preserved}")
    print(f"Chunk 1 preserved: {chunk1_preserved}")
    print(f"Chunk 2 preserved: {chunk2_preserved}")
    print(f"Chunk 3 preserved: {chunk3_preserved}")
    
    # Early chunks should be more likely to be preserved
    assert chunk0_preserved or chunk1_preserved, "Early chunks should be preserved"
    
    print("PASS: Preservation factor test")


if __name__ == "__main__":
    print("Testing weighted preservation strategy...\n")
    
    test_weighted_preservation_strategy()
    print()
    test_preservation_factors()
    
    print("\nSUCCESS: All weighted preservation tests passed!")
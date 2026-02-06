import re
import hashlib
from pathlib import Path

def get_file_hash(file_path: Path) -> str:
    """Calculate the MD5 hash of a file to uniquely identify its content."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def sanitize_filename(filename: str, max_length: int = 150) -> str:
    """
    Replace invalid characters and remove all control characters.
    Limits length to stay within Windows path limits.
    """
    # Remove invalid characters for Windows paths: < > : " / \ | ? *
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    
    # Remove ALL control characters (including \r, \n, \t) and non-printable chars
    sanitized = "".join(char for char in sanitized if char.isprintable())
    
    # Replace any multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # Remove consecutive dots
    sanitized = re.sub(r'\.\.+', '.', sanitized)
    
    # Trim leading/trailing spaces and dots
    sanitized = sanitized.strip(" .")
    
    # Limit length (preserving extension if possible)
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].strip()
        
    return sanitized

def proportional_chunk_trim(text: str, target_length: int, num_chunks: int) -> str:
    """
    Trim text to a target length by dividing it into equal chunks and applying a reduction factor.
    """
    if not text:
        return ""
        
    # Calculate the size of each chunk
    chunk_size = len(text) // num_chunks if num_chunks > 0 else len(text)
    if chunk_size == 0:
        return text[:target_length]
    
    # Divide the text into equal chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Calculate the reduction factor
    total_length = sum(len(chunk) for chunk in chunks)
    reduction_factor = min(1.0, target_length / total_length) if total_length > 0 else 0
    
    # Apply the reduction factor to each chunk
    trimmed_text = ""
    current_length = 0
    for chunk in chunks:
        allowed_length = int(len(chunk) * reduction_factor)
        trimmed_chunk = chunk[:allowed_length]
        trimmed_text += trimmed_chunk
        current_length += len(trimmed_chunk)
        
        # Stop if we've reached the target length
        if current_length >= target_length:
            break
    
    return trimmed_text

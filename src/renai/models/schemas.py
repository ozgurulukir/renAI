from typing import Optional
from pydantic import BaseModel, Field


class BaseMetadata(BaseModel):
    """Base schema for all metadata extracted by renAI."""
    
    language: str = Field(
        ..., 
        description="ISO 639-1 language code (e.g., 'tr' for Turkish, 'en' for English/Other)."
    )
    category: str = Field(
        ..., 
        description="The most appropriate category chosen from the allowed list."
    )

    @property
    def safe_category(self) -> str:
        """Return a filesystem-safe version of the category."""
        if not self.category:
            return "Uncategorized"
        # Basic sanitization for categories (replace slashes)
        safe = self.category.replace("/", "&").replace("\\", "&")
        # Note: final character sanitization still happens in builder/utils 
        # to avoid circular imports here.
        return safe.strip() or "Uncategorized"


class BookMetadata(BaseMetadata):
    """Metadata schema for documents (PDF, EPUB, etc.)."""
    
    title: str = Field(
        ..., 
        description="The full, official title of the document. Do not truncate."
    )
    subtitle: Optional[str] = Field(
        None, 
        description="The subtitle of the document, if present."
    )
    edition: Optional[str] = Field(
        None, 
        description="The edition information (e.g., '2nd Edition', 'Revised')."
    )
    author: Optional[str] = Field(
        None, 
        description="The full name of the author(s) or editor(s)."
    )
    year: Optional[int] = Field(
        None, 
        description="The 4-digit publication year."
    )


class ImageMetadata(BaseMetadata):
    """Metadata schema for image files analyzed via vision."""
    
    title: Optional[str] = Field(
        None, 
        description="A short title for the image, if applicable."
    )
    description: str = Field(
        ..., 
        description="A concise, descriptive filename-ready summary of the image content."
    )
    visual_context: Optional[str] = Field(
        None, 
        description="A more detailed 1-2 sentence description of the visual scene."
    )
    exif_date: Optional[str] = Field(
        None, 
        description="The date extracted from EXIF (internal use only, usually provided by processor)."
    )

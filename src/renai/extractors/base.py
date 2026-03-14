from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

class BaseExtractor(ABC):
    """Base class for all text extractors."""
    
    @abstractmethod
    def extract_text(self, file_path: Path) -> Optional[str]:
        """Extract text from the given file."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the extractor."""
        pass
